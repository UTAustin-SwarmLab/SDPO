#!/usr/bin/env python3
"""
Probe whether a model leaks privileged context when conditioned on a correct prior attempt.

Workflow
--------
1. Run N independent rollouts per SciKnowEval question (default N=4).
2. Reprompt with privileged context and regenerate. Mode controls what is shown:
   - ground_truth: always use the dataset answer as privileged context.
   - mixed: use a correct initial rollout if any; otherwise fall back to ground truth.
3. A judge LLM (e.g. GPT-5.2) scores whether the regenerated reasoning jumps
   directly to the answer without genuine sequential derivation — a sign of leakage.

Usage
-----
# Inference via a local vLLM OpenAI-compatible server; judge via OpenAI API:
python probes/leak_test.py \\
    --mode ground_truth \\
    --dataset-path datasets/sciknoweval/physics/test.json \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --model-base-url http://localhost:8000/v1 \\
    --judge-model gpt-5.2 \\
    --output probes/results/leak_physics_gt.jsonl

python probes/leak_test.py \\
    --mode ground_truth \\
    --seed 42 --max-examples 100 \\
    ...

python probes/leak_test.py \\
    --mode mixed \\
    --seed 42 --max-examples 100 \\
    ...

# Or pin the exact indices from the first run:
python probes/leak_test.py \\
    --mode mixed \\
    --sample-indices-file probes/results/leak_physics_gt.samples.json \\
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
from typing import Any, Literal, Optional

from openai import OpenAI

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl.utils.reward_score.feedback.mcq import compute_score as mcq_score

# SDPO self-distillation reprompt templates (verl/workers/config/actor.py)
REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n\n"
    "Correctly solve the original question.\n"
)
SOLUTION_TEMPLATE = (
    "\n"
    "Correct solution:\n\n"
    "{successful_previous_attempt}\n\n"
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of reasoning quality in multiple-choice science questions.

Your task is to detect **reasoning leakage**: when a model was given a privileged \
correct prior attempt and then asked to solve the question again, does its new \
reasoning genuinely derive the answer step-by-step, or does it **jump directly** \
to the conclusion with insufficient intermediate steps?

Signs of leakage (any of these suggests leakage):
- The reasoning skips key derivation steps that a competent solver would need.
- The reasoning mirrors or closely paraphrases the privileged solution without \
  independent derivation.
- The answer appears early with minimal justification, as if the model already \
  knew the result.
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


PrivilegedMode = Literal["ground_truth", "mixed"]


@dataclass
class LeakProbeResult:
    example: Example
    mode: PrivilegedMode
    initial_rollouts: list[RolloutResult]
    any_initial_correct: bool
    privileged_solution: Optional[str] = None
    privileged_source: Optional[str] = None  # "model_rollout" | "ground_truth"
    reprompt_response: Optional[str] = None
    reprompt_correct: Optional[bool] = None
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
    """Select a fixed subset so ground_truth and mixed runs analyze the same questions."""
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
    """Deterministic per-call seed for inference (fits 32-bit API limits)."""
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
    )


def remove_thinking_trace(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def build_reprompt_user(prompt: str, successful_attempt: str, remove_thinking: bool) -> str:
    solution = successful_attempt
    if remove_thinking:
        solution = remove_thinking_trace(solution)
    solution_section = SOLUTION_TEMPLATE.format(successful_previous_attempt=solution)
    return REPROMPT_TEMPLATE.format(prompt=prompt, solution=solution_section, feedback="")


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

Privileged correct prior attempt (shown to the model before regeneration):
{privileged_solution}

Regenerated response to evaluate:
{regenerated_response}

Does the regenerated reasoning show leakage?\
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


def run_initial_rollouts(
    client: InferenceClient,
    example: Example,
    num_samples: int,
) -> list[RolloutResult]:
    messages = build_messages(example.system, example.prompt)
    rollouts: list[RolloutResult] = []
    for sample_idx in range(num_samples):
        response = client.generate(messages, example_idx=example.idx, call_index=sample_idx)
        rollouts.append(score_response(response, example.answer))
    return rollouts


def pick_privileged_solution(
    rollouts: list[RolloutResult],
    ground_truth: str,
    prompt: str,
    mode: PrivilegedMode,
) -> tuple[str, str]:
    """Return (privileged_solution, source) where source is model_rollout or ground_truth."""
    if mode == "ground_truth":
        return format_ground_truth_solution(ground_truth, prompt), "ground_truth"

    for rollout in rollouts:
        if rollout.correct:
            return rollout.response, "model_rollout"
    return format_ground_truth_solution(ground_truth, prompt), "ground_truth"


def extract_option_text(prompt: str, answer: str) -> Optional[str]:
    """Extract the text following an MCQ option label from the prompt."""
    match = re.search(rf"^{re.escape(answer)}:\s*(.+)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def format_ground_truth_solution(answer: str, prompt: str) -> str:
    """Correct response in SciKnowEval MCQ format with explicit option text."""
    option_text = extract_option_text(prompt, answer)
    if option_text:
        reasoning = f"The answer to the question is Option {answer}. {option_text}"
    else:
        reasoning = f"The answer to the question is Option {answer}."
    return f"Use the following information to solve the question: {reasoning}"


def probe_example(
    infer_client: InferenceClient,
    judge_client: Optional[JudgeClient],
    example: Example,
    mode: PrivilegedMode,
    num_samples: int,
    remove_thinking: bool,
) -> LeakProbeResult:
    if mode == "mixed":
        initial = run_initial_rollouts(infer_client, example, num_samples)
    else:
        initial = []

    any_correct = any(r.correct for r in initial)

    privileged, privileged_source = pick_privileged_solution(initial, example.answer, example.prompt, mode)
    reprompt_user = build_reprompt_user(example.prompt, privileged, remove_thinking)
    reprompt_messages = build_messages(example.system, reprompt_user)
    reprompt_response = infer_client.generate(
        reprompt_messages,
        example_idx=example.idx,
        call_index=num_samples,  # distinct from initial rollout seeds
    )
    reprompt_scored = score_response(reprompt_response, example.answer)

    judge_result: Optional[dict[str, Any]] = None
    if judge_client is not None:
        judge_result = judge_client.evaluate(
            question=example.prompt,
            privileged_solution=privileged,
            regenerated_response=reprompt_response,
            ground_truth=example.answer,
        )

    return LeakProbeResult(
        example=example,
        mode=mode,
        initial_rollouts=initial,
        any_initial_correct=any_correct,
        privileged_solution=privileged,
        privileged_source=privileged_source,
        reprompt_response=reprompt_response,
        reprompt_correct=reprompt_scored.correct,
        judge=judge_result,
    )


def serialize_result(result: LeakProbeResult) -> dict[str, Any]:
    initial_accuracy = (
        sum(r.correct for r in result.initial_rollouts) / len(result.initial_rollouts)
        if result.initial_rollouts
        else None
    )
    return {
        "idx": result.example.idx,
        "dataset": result.example.dataset,
        "mode": result.mode,
        "answer": result.example.answer,
        "any_initial_correct": result.any_initial_correct,
        "privileged_source": result.privileged_source,
        "initial_rollouts": [asdict(r) for r in result.initial_rollouts],
        "initial_accuracy": initial_accuracy,
        "privileged_solution": result.privileged_solution,
        "reprompt_response": result.reprompt_response,
        "reprompt_correct": result.reprompt_correct,
        "judge": result.judge,
    }


def summarize(results: list[LeakProbeResult], mode: PrivilegedMode, sample_indices: list[int], seed: int) -> dict[str, Any]:
    judged = [r for r in results if r.judge and r.judge.get("leak_detected") is not None]
    from_rollout = [r for r in results if r.privileged_source == "model_rollout"]
    from_ground_truth = [r for r in results if r.privileged_source == "ground_truth"]
    rollout_totals = sum(len(res.initial_rollouts) for res in results)

    leak_count = sum(1 for r in judged if r.judge.get("leak_detected"))
    jump_count = sum(1 for r in judged if r.judge.get("jumps_to_answer"))
    mirror_count = sum(1 for r in judged if r.judge.get("mirrors_privileged_solution"))

    return {
        "mode": mode,
        "seed": seed,
        "sample_indices": sample_indices,
        "total_examples": len(results),
        "privileged_from_model_rollout": len(from_rollout),
        "privileged_from_ground_truth": len(from_ground_truth),
        "judged_examples": len(judged),
        "leak_rate": leak_count / len(judged) if judged else None,
        "jumps_to_answer_rate": jump_count / len(judged) if judged else None,
        "mirrors_privileged_rate": mirror_count / len(judged) if judged else None,
        "mean_initial_accuracy": (
            sum(sum(r.correct for r in res.initial_rollouts) for res in results) / rollout_totals
            if rollout_totals
            else None
        ),
        "reprompt_accuracy": (
            sum(1 for r in results if r.reprompt_correct) / len(results) if results else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe privileged-context reasoning leakage on SciKnowEval.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/sciknoweval/physics/test.json",
        help="Path to SciKnowEval JSONL (train.json or test.json).",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name/path for inference.")
    parser.add_argument(
        "--model-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for inference (e.g. http://localhost:8000/v1). "
        "Defaults to OpenAI if unset.",
    )
    parser.add_argument("--model-api-key", type=str, default=None, help="API key for inference client.")
    parser.add_argument(
        "--mode",
        choices=["ground_truth", "mixed"],
        default="ground_truth",
        help="Privileged context mode. ground_truth: always use GT answer. "
        "mixed: use a correct initial rollout when available, else GT.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Initial rollouts per question (mixed mode only).",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None, help="Randomly sample this many examples (requires --seed).")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for example selection and inference. Use the same seed across modes.",
    )
    parser.add_argument(
        "--sample-indices-file",
        type=str,
        default=None,
        help="JSON file with {\"indices\": [...]} from a prior run. Overrides --seed/--max-examples selection.",
    )
    parser.add_argument(
        "--remove-thinking-from-demonstration",
        action="store_true",
        help="Strip <think> tags from privileged solution before reprompting.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-5.2", help="Judge model (e.g. gpt-5.2).")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Optional separate base URL for judge.")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for judge (defaults to OPENAI_API_KEY).")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge evaluation (inference only).")
    parser.add_argument("--output", type=str, default="probes/results/leak_test.jsonl")
    parser.add_argument("--workers", type=int, default=1, help="Parallel examples (use 1 for local vLLM).")
    parser.add_argument("--sleep-between", type=float, default=0.0, help="Seconds to sleep between examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    results: list[LeakProbeResult] = []

    def _run_one(ex: Example) -> LeakProbeResult:
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)
        return probe_example(
            infer_client=infer_client,
            judge_client=judge_client,
            example=ex,
            mode=args.mode,
            num_samples=args.num_samples,
            remove_thinking=args.remove_thinking_from_demonstration,
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

    # Preserve dataset order in output.
    results.sort(key=lambda r: r.example.idx)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(serialize_result(result)) + "\n")

    summary = summarize(results, mode=args.mode, sample_indices=sample_indices, seed=args.seed)
    summary_path = output_path.with_suffix(".summary.json")
    samples_path = output_path.with_suffix(".samples.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(samples_path, "w") as f:
        json.dump({"seed": args.seed, "indices": sample_indices}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-example results to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote sample indices to {samples_path}")


if __name__ == "__main__":
    main()
