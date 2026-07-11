#!/usr/bin/env python3
"""
Test true in-context learning on SciKnowEval MCQ questions.

Workflow
--------
1. Sample one rollout per question.
2. Score with the MCQ verifier.
3. If incorrect, reprompt with the previous attempt and feedback (no ground truth).
4. Skip regeneration when the first rollout is already correct.
5. Optionally judge ICL second-pass responses for privileged-solution leakage
   (same metric as leak_test: does the GT answer appear in the reasoning trace?).

Usage
-----
python probes/icl_test.py \\
    --dataset-path datasets/sciknoweval/physics/test.json \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --model-base-url http://localhost:8000/v1 \\
    --judge-model gpt-5.2 \\
    --seed 42 --max-examples 100 \\
    --output probes/results/icl_physics.jsonl

# Reuse the same question indices as a prior leak probe run:
python probes/icl_test.py \\
    --sample-indices-file probes/results/leak_physics.samples.json \\
    ...

Requires OPENAI_API_KEY for the judge (and for the model if not using a local server).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl.utils.reward_score.feedback.mcq import compute_score as mcq_score

DEFAULT_CORRECT_FEEDBACK = "Your previous answer was correct."
DEFAULT_INCORRECT_FEEDBACK = "Your previous answer was incorrect."
DEFAULT_PROMPT_TEMPLATE = (
    "{prompt}\n\n"
    "Previous answer:\n{previous_answer}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Answer the original question again."
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of reasoning quality in multiple-choice science questions.

Your task is to detect **reasoning leakage**: compare the regenerated response against the \
privileged correct solution below. In this in-context learning setting, the model was NOT \
shown the privileged solution — it only saw its previous incorrect attempt and feedback. \
Does the regenerated reasoning nevertheless mirror or leak this privileged correct \
solution (e.g., step-skipping, jumping to the answer, closely paraphrasing the option text)?

Signs of leakage (any of these suggests leakage):
- The reasoning skips key derivation steps that a competent solver would need.
- The reasoning mirrors or closely paraphrases the privileged solution without \
  independent derivation.
- The answer appears early with minimal justification, as if the model already \
  knew the result from the privileged solution.
- Reasoning is suspiciously short or generic given the question difficulty.

Signs of genuine reasoning (suggests NO leakage):
- Clear sequential steps from problem setup to conclusion.
- Intermediate calculations or logical steps appropriate to the domain.
- Reasoning that could stand alone without the privileged context.

Respond with a JSON object only (no markdown fences):
{
  "leak_detected": <bool>,
  "jumps_to_answer": <bool>,
  "mirrors_privileged_solution": <bool>,
  "reasoning_quality": <int 1-5, 5=fully rigorous>,
  "explanation": "<one short paragraph>"
}
"""


@dataclass
class Example:
    idx: int
    prompt: str
    system: str
    answer: str
    dataset: str = "sciknoweval"


@dataclass
class RolloutResult:
    response: str
    correct: bool
    pred: Optional[str]
    incorrect_format: bool
    feedback: str = ""


@dataclass
class ICLProbeResult:
    example: Example
    first_rollout: RolloutResult
    icl_attempted: bool
    icl_feedback: Optional[str] = None
    icl_response: Optional[str] = None
    icl_correct: Optional[bool] = None
    judge: Optional[dict[str, Any]] = None


def load_jsonl_dataset(path: str) -> list[Example]:
    examples: list[Example] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    idx=row["idx"],
                    prompt=row["prompt"],
                    system=row.get("system", "").strip(),
                    answer=row["answer"],
                    dataset=row.get("dataset", "sciknoweval"),
                )
            )
    return examples


def select_examples(
    examples: list[Example],
    *,
    max_examples: Optional[int],
    seed: int,
    sample_indices_file: Optional[str],
) -> tuple[list[Example], list[int]]:
    by_idx = {ex.idx: ex for ex in examples}
    all_indices = sorted(by_idx)

    if sample_indices_file:
        with open(sample_indices_file) as f:
            payload = json.load(f)
        selected_indices = payload["indices"] if isinstance(payload, dict) else payload
        missing = [idx for idx in selected_indices if idx not in by_idx]
        if missing:
            raise SystemExit(f"Indices not found in dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    elif max_examples is not None and max_examples < len(examples):
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(all_indices, max_examples))
    else:
        selected_indices = all_indices

    return [by_idx[idx] for idx in selected_indices], selected_indices


def inference_seed(base_seed: int, example_idx: int, call_index: int) -> int:
    return (base_seed + example_idx * 10_000 + call_index) % (2**31 - 1)


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def score_response(response: str, ground_truth: str) -> RolloutResult:
    result = mcq_score(response, ground_truth)
    return RolloutResult(
        response=response,
        correct=bool(result["acc"]),
        pred=result.get("pred"),
        incorrect_format=bool(result.get("incorrect_format")),
        feedback=result.get("feedback", ""),
    )


def remove_thinking_trace(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def build_icl_feedback(
    first_rollout: RolloutResult,
    *,
    correct_feedback: str,
    incorrect_feedback: str,
    include_environment_feedback: bool,
) -> str:
    if first_rollout.correct:
        feedback = correct_feedback
    else:
        feedback = incorrect_feedback

    env_feedback = first_rollout.feedback.strip()
    if include_environment_feedback and env_feedback:
        feedback = f"{feedback}\n\nAdditional feedback:\n{env_feedback}"
    return feedback


def build_icl_user(
    prompt: str,
    previous_answer: str,
    feedback: str,
    prompt_template: str,
    *,
    remove_thinking: bool,
) -> str:
    previous = previous_answer
    if remove_thinking:
        previous = remove_thinking_trace(previous)
    return prompt_template.format(
        prompt=prompt,
        previous_answer=previous,
        feedback=feedback,
    )


def extract_option_text(prompt: str, answer: str) -> Optional[str]:
    match = re.search(rf"^{re.escape(answer)}:\s*(.+)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def format_ground_truth_solution(answer: str, prompt: str) -> str:
    """Privileged correct solution reference (same format as leak_test ground_truth mode)."""
    option_text = extract_option_text(prompt, answer)
    if option_text:
        reasoning = f"The answer to the question is Option {answer}. {option_text}"
    else:
        reasoning = f"The answer to the question is Option {answer}."
    return f"Use the following information to solve the question: {reasoning}"


class InferenceClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
        )

    def generate(self, messages: list[dict[str, str]], *, example_idx: int, call_index: int) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=inference_seed(self.seed, example_idx, call_index),
        )
        return completion.choices[0].message.content or ""


class JudgeClient:
    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str]):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def evaluate(
        self,
        question: str,
        privileged_solution: str,
        regenerated_response: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        user_content = f"""\
Question:
{question}

Ground-truth answer: {ground_truth}

Privileged correct solution (NOT shown to the model; reference for leakage detection):
{privileged_solution}

Regenerated response to evaluate:
{regenerated_response}

Does the regenerated reasoning mirror or leak this privileged correct solution?\
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "leak_detected": None,
                "parse_error": True,
                "raw_judge_output": raw,
            }


def probe_example(
    infer_client: InferenceClient,
    judge_client: Optional[JudgeClient],
    example: Example,
    *,
    correct_feedback: str,
    incorrect_feedback: str,
    include_environment_feedback: bool,
    prompt_template: str,
    remove_thinking: bool,
) -> ICLProbeResult:
    messages = build_messages(example.system, example.prompt)
    first_response = infer_client.generate(messages, example_idx=example.idx, call_index=0)
    first_rollout = score_response(first_response, example.answer)

    if first_rollout.correct:
        return ICLProbeResult(
            example=example,
            first_rollout=first_rollout,
            icl_attempted=False,
        )

    icl_feedback = build_icl_feedback(
        first_rollout,
        correct_feedback=correct_feedback,
        incorrect_feedback=incorrect_feedback,
        include_environment_feedback=include_environment_feedback,
    )
    icl_user = build_icl_user(
        example.prompt,
        first_rollout.response,
        icl_feedback,
        prompt_template,
        remove_thinking=remove_thinking,
    )
    icl_messages = build_messages(example.system, icl_user)
    icl_response = infer_client.generate(icl_messages, example_idx=example.idx, call_index=1)
    icl_scored = score_response(icl_response, example.answer)

    judge_result: Optional[dict[str, Any]] = None
    if judge_client is not None:
        privileged_solution = format_ground_truth_solution(example.answer, example.prompt)
        judge_result = judge_client.evaluate(
            question=example.prompt,
            privileged_solution=privileged_solution,
            regenerated_response=icl_response,
            ground_truth=example.answer,
        )

    return ICLProbeResult(
        example=example,
        first_rollout=first_rollout,
        icl_attempted=True,
        icl_feedback=icl_feedback,
        icl_response=icl_response,
        icl_correct=icl_scored.correct,
        judge=judge_result,
    )


def serialize_result(result: ICLProbeResult) -> dict[str, Any]:
    return {
        "idx": result.example.idx,
        "dataset": result.example.dataset,
        "answer": result.example.answer,
        "first_rollout": asdict(result.first_rollout),
        "first_correct": result.first_rollout.correct,
        "icl_attempted": result.icl_attempted,
        "icl_feedback": result.icl_feedback,
        "icl_response": result.icl_response,
        "icl_correct": result.icl_correct,
        "final_correct": (
            result.first_rollout.correct
            if not result.icl_attempted
            else result.icl_correct
        ),
        "privileged_solution": (
            format_ground_truth_solution(result.example.answer, result.example.prompt)
            if result.icl_attempted
            else None
        ),
        "judge": result.judge,
    }


def summarize(results: list[ICLProbeResult], sample_indices: list[int], seed: int) -> dict[str, Any]:
    total = len(results)
    first_correct = [r for r in results if r.first_rollout.correct]
    first_incorrect = [r for r in results if not r.first_rollout.correct]
    icl_attempted = [r for r in results if r.icl_attempted]
    icl_recovered = [r for r in icl_attempted if r.icl_correct]
    icl_failed = [r for r in icl_attempted if not r.icl_correct]

    pass_at_1 = len(first_correct) / total if total else None
    recovery_rate = len(icl_recovered) / len(icl_attempted) if icl_attempted else None
    icl_accuracy = len(icl_recovered) / len(icl_attempted) if icl_attempted else None
    final_accuracy = (
        (len(first_correct) + len(icl_recovered)) / total if total else None
    )

    judged = [
        r for r in icl_attempted
        if r.judge and r.judge.get("leak_detected") is not None
    ]
    leak_count = sum(1 for r in judged if r.judge.get("leak_detected"))
    jump_count = sum(1 for r in judged if r.judge.get("jumps_to_answer"))
    mirror_count = sum(
        1 for r in judged
        if r.judge.get("mirrors_privileged_solution") or r.judge.get("mirrors_previous_attempt")
    )

    return {
        "seed": seed,
        "sample_indices": sample_indices,
        "total_examples": total,
        "pass_at_1": pass_at_1,
        "first_correct_count": len(first_correct),
        "first_incorrect_count": len(first_incorrect),
        "icl_attempted_count": len(icl_attempted),
        "icl_skipped_count": len(first_correct),
        "icl_accuracy": icl_accuracy,
        "recovery_rate": recovery_rate,
        "icl_recovered_count": len(icl_recovered),
        "icl_failed_count": len(icl_failed),
        "final_accuracy": final_accuracy,
        "judged_examples": len(judged),
        "leak_rate": leak_count / len(judged) if judged else None,
        "jumps_to_answer_rate": jump_count / len(judged) if judged else None,
        "mirrors_privileged_rate": mirror_count / len(judged) if judged else None,
    }


def load_results_from_jsonl(path: str, examples_by_idx: dict[int, Example]) -> list[ICLProbeResult]:
    results: list[ICLProbeResult] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            example = examples_by_idx[row["idx"]]
            first = row["first_rollout"]
            results.append(
                ICLProbeResult(
                    example=example,
                    first_rollout=RolloutResult(
                        response=first["response"],
                        correct=first["correct"],
                        pred=first.get("pred"),
                        incorrect_format=first["incorrect_format"],
                        feedback=first.get("feedback", ""),
                    ),
                    icl_attempted=row["icl_attempted"],
                    icl_feedback=row.get("icl_feedback"),
                    icl_response=row.get("icl_response"),
                    icl_correct=row.get("icl_correct"),
                    judge=row.get("judge"),
                )
            )
    return results


def judge_existing_results(
    results: list[ICLProbeResult],
    judge_client: JudgeClient,
) -> list[ICLProbeResult]:
    judged_results: list[ICLProbeResult] = []
    icl_attempted = [r for r in results if r.icl_attempted]
    for i, result in enumerate(icl_attempted, start=1):
        print(f"[judge {i}/{len(icl_attempted)}] idx={result.example.idx}", flush=True)
        privileged_solution = format_ground_truth_solution(result.example.answer, result.example.prompt)
        judge_result = judge_client.evaluate(
            question=result.example.prompt,
            privileged_solution=privileged_solution,
            regenerated_response=result.icl_response or "",
            ground_truth=result.example.answer,
        )
        judged_results.append(
            ICLProbeResult(
                example=result.example,
                first_rollout=result.first_rollout,
                icl_attempted=result.icl_attempted,
                icl_feedback=result.icl_feedback,
                icl_response=result.icl_response,
                icl_correct=result.icl_correct,
                judge=judge_result,
            )
        )

    judged_by_idx = {r.example.idx: r for r in judged_results}
    merged: list[ICLProbeResult] = []
    for result in results:
        if result.example.idx in judged_by_idx:
            merged.append(judged_by_idx[result.example.idx])
        else:
            merged.append(result)
    merged.sort(key=lambda r: r.example.idx)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test true in-context learning on SciKnowEval.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/sciknoweval/physics/test.json",
        help="Path to SciKnowEval JSONL (train.json or test.json).",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name/path for inference.")
    parser.add_argument(
        "--model-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for inference (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument("--model-api-key", type=str, default=None, help="API key for inference client.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-indices-file",
        type=str,
        default=None,
        help="JSON file with {\"indices\": [...]} from a prior run.",
    )
    parser.add_argument(
        "--include-environment-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append MCQ verifier feedback to the ICL reprompt.",
    )
    parser.add_argument(
        "--correct-feedback",
        type=str,
        default=DEFAULT_CORRECT_FEEDBACK,
    )
    parser.add_argument(
        "--incorrect-feedback",
        type=str,
        default=DEFAULT_INCORRECT_FEEDBACK,
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Reprompt template with {prompt}, {previous_answer}, {feedback}.",
    )
    parser.add_argument(
        "--remove-thinking-from-previous-answer",
        action="store_true",
        help="Strip  tags from the shown previous attempt.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-5.2", help="Judge model (e.g. gpt-5.2).")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Optional separate base URL for judge.")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for judge (defaults to OPENAI_API_KEY).")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge evaluation (inference only).")
    parser.add_argument(
        "--judge-only",
        type=str,
        default=None,
        help="Skip inference and judge an existing ICL results JSONL.",
    )
    parser.add_argument("--output", type=str, default="probes/results/icl_test.jsonl")
    parser.add_argument("--workers", type=int, default=1, help="Parallel examples (use 1 for local vLLM).")
    parser.add_argument("--sleep-between", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.judge_only:
        if args.skip_judge:
            raise SystemExit("--judge-only cannot be combined with --skip-judge")
        input_path = Path(args.judge_only)
        samples_path = input_path.with_suffix(".samples.json")
        if not samples_path.exists():
            raise SystemExit(f"Missing samples file: {samples_path}")
        with open(samples_path) as f:
            samples_payload = json.load(f)
        sample_indices = samples_payload["indices"]
        seed = samples_payload.get("seed", args.seed)

        all_examples = load_jsonl_dataset(args.dataset_path)
        examples_by_idx = {ex.idx: ex for ex in all_examples}
        results = load_results_from_jsonl(str(input_path), examples_by_idx)
        judge_client = JudgeClient(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
        )
        results = judge_existing_results(results, judge_client)
        output_path = input_path
    else:
        if not args.model:
            raise SystemExit("--model is required unless using --judge-only")

        all_examples = load_jsonl_dataset(args.dataset_path)
        if not all_examples:
            raise SystemExit(f"No examples loaded from {args.dataset_path}")

        examples, sample_indices = select_examples(
            all_examples,
            max_examples=args.max_examples,
            seed=args.seed,
            sample_indices_file=args.sample_indices_file,
        )
        if not examples:
            raise SystemExit("No examples selected.")

        print(f"Selected {len(examples)} examples (seed={args.seed})", flush=True)
        print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}", flush=True)

        infer_client = InferenceClient(
            model=args.model,
            base_url=args.model_base_url,
            api_key=args.model_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

        judge_client: Optional[JudgeClient] = None
        if not args.skip_judge:
            judge_client = JudgeClient(
                model=args.judge_model,
                api_key=args.judge_api_key,
                base_url=args.judge_base_url,
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results: list[ICLProbeResult] = []

        def _run_one(ex: Example) -> ICLProbeResult:
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
            return probe_example(
                infer_client=infer_client,
                judge_client=judge_client,
                example=ex,
                correct_feedback=args.correct_feedback,
                incorrect_feedback=args.incorrect_feedback,
                include_environment_feedback=args.include_environment_feedback,
                prompt_template=args.prompt_template,
                remove_thinking=args.remove_thinking_from_previous_answer,
            )

        if args.workers <= 1:
            for i, ex in enumerate(examples):
                print(f"[{i + 1}/{len(examples)}] idx={ex.idx}", flush=True)
                results.append(_run_one(ex))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_one, ex): ex for ex in examples}
                for i, future in enumerate(as_completed(futures), start=1):
                    ex = futures[future]
                    print(f"[{i}/{len(examples)}] idx={ex.idx} done", flush=True)
                    results.append(future.result())

        results.sort(key=lambda r: r.example.idx)
        seed = args.seed

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(serialize_result(result)) + "\n")

    summary = summarize(results, sample_indices=sample_indices, seed=seed)
    summary_path = output_path.with_suffix(".summary.json")
    samples_path = output_path.with_suffix(".samples.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(samples_path, "w") as f:
        json.dump({"seed": seed, "indices": sample_indices}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-example results to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote sample indices to {samples_path}")


if __name__ == "__main__":
    main()
