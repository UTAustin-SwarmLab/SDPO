# ICML 2027 experiment runbook

This file tells a Claude Code agent (and the person it works with) two things: which runs make up the ICML 2027 comparison, and which hyperparameters must stay the same across them. There are 28 runs over five algorithms, all from scratch. Anything not fixed here, such as paths, partition, account or launcher, may be adapted to your machine. The exact submit commands are in "Commands" near the end.

Ask the user before anything costly or destructive: submitting jobs, cancelling jobs, deleting checkpoints, editing tracked files, committing or pushing.

## What the experiments test

Every algorithm trains Qwen3-8B on SciKnowEval (biology, chemistry, material and physics together) with the same data, batch sizes, learning rate, step count and validation, so the algorithms can be compared directly.

- **SDQL** regresses a Q-function (the student's log-prob ratio over the top-100 tokens) onto per-token rewards from an EMA self-teacher. Its 18 runs ask:
  1. **Target:** does a λ-return target (λ = 0.9 or 1.0) beat the plain on-policy one-step target?
  2. **Off-policyness:** does splitting each step's batch into 4 optimizer updates (mini-batch 8) instead of 1 (mini-batch 32) help or hurt?
  3. **Reward shape:** α ∈ {0, 0.5, 1}, from forward-KL-style to reverse-KL-style.
- **Baselines:**
  - **GRPO**: RL from the verifiable reward with the standard group-normalised advantage. It also gets the off-policy arm and lr 3e-6.
  - **ROVER**: Q regression onto the GRPO advantage. It also gets the off-policy arm.
  - **Distil**: distillation from the EMA teacher at α = 1.
  - **SDPO**: plain stepwise self-distillation from the EMA teacher at α ∈ {0, 0.5, 1}.

Only GRPO, ROVER and SDQL have off-policy (mini-batch 8) runs, and only GRPO has a second learning rate. That asymmetry is intended.

## The 28 runs

| Algorithm | Config | Job script | Runs | What varies |
|---|---|---|---|---|
| GRPO | `baseline_grpo` | `tacc/jobs/run_grpo.slurm` | 4 | mini-batch {32, 8} × lr {1e-5, 3e-6} |
| ROVER | `rover` | `tacc/jobs/run_rover.slurm` | 2 | mini-batch {32, 8} |
| Distil | `distil` | `tacc/jobs/run_distil.slurm` | 1 | nothing (α = 1) |
| SDPO | `sdpo` | `tacc/jobs/run_sdpo.slurm` | 3 | α {0, 0.5, 1} |
| SDQL | `sdql` | `tacc/jobs/run_sdql.slurm` | 18 | mini-batch {32, 8} × target {on-policy, λ = 0.9, λ = 1.0} × α {0, 0.5, 1} |

Each run's name is the last part of its command in "Commands". Do not rename runs: results are compared by name, and the name is also the checkpoint folder that auto-resume uses.

## Hyperparameters that must stay the same

These are already the defaults in the repo; the commands do not need to set them. Do not change them.

| Setting | Value | Where it is set |
|---|---|---|
| Model | `Qwen/Qwen3-8B` | job scripts, `verl/trainer/config/user.yaml` |
| Data | `datasets/sciknoweval2/all/` | first argument of every command |
| Prompts per step | 32 | `data.train_batch_size` |
| Answers per prompt | 8 | `actor_rollout_ref.rollout.n` |
| Mini-batch | 32 (one optimizer update per step); 8 only in the off-policy runs | `actor.ppo_mini_batch_size` |
| Learning rate | 1e-5 with 10 warm-up steps (GRPO also 3e-6) | `actor.optim.lr`, `actor.optim.lr_warmup_steps` |
| Training length | 400 steps | `trainer.total_training_steps` |
| Lengths | prompt 2048, answer 6144 tokens | `data.max_prompt_length`, `data.max_response_length` |
| Sampling | temperature 1.0 in training | `rollout.temperature` |
| Validation | at step 0 and every 20 steps; 16 samples per question at temperature 0.7, top-p 0.95 | `trainer.val_before_train`, `trainer.test_freq`, `rollout.val_kwargs` |
| ICL validation | off | `trainer.icl_validation.enable` |
| Checkpoints | every 20 steps | `trainer.save_freq` |
| Advantage | GRPO, normalised by the group std | `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True` |
| Rollout correction | token-level, threshold 2.0 | `algorithm.rollout_correction` |
| Policy clip and regularisers | clip 0.2 / 0.28, no KL loss, no entropy bonus, grad clip 1.0 | `actor.*` |
| Distillation logits | full logits, top-100 | `self_distillation.full_logit_distillation`, `distillation_topk` |
| W&B project | `sdpo20icml2027` | `trainer.project_name` |
| GPUs | 8 per run | `trainer.nnodes × trainer.n_gpus_per_node = 8` |

## Settings that differ by design

| Setting | GRPO | ROVER | Distil | SDPO | SDQL |
|---|---|---|---|---|---|
| Teacher EMA rate | – | – | 0.01 | 0.01 | 0.01 |
| α | – | – | 1 | 0, 0.5, 1 | 0, 0.5, 1 |
| Importance-sampling clip in the loss | – | off | 2.0 | 2.0 | off |
| Reward clamp (±2) | – | – | on | on | off |
| Tail-mass term | – | – | on | on | off |
| Regression target | – | GRPO advantage + mean next-token Q | – | – | on-policy one-step or λ-return |
| Future-return terms | – | – | – | off (CISPO clip 0.99 and the future-returns baseline are inactive) | – |

## Micro-batch: the one setting you may change

The micro-batch per GPU is **8** by default (`actor.ppo_micro_batch_size_per_gpu`). It only controls how many sequences go through the GPU at once, so it changes memory use and speed, not the update.

If a run crashes with a GPU out-of-memory error, resubmit **that run only** with micro-batch **2** (see "Commands", step 3). It resumes from its last checkpoint. Leave every other run at 8, and list the runs that needed 2 in your results report.

## Setup

1. **Code.** Clone `https://github.com/UTAustin-SwarmLab/SDPO` (branch `main`). Check that the checkout has this experiment's settings:
   ```bash
   grep -c "norm_adv_by_std_in_grpo: True" verl/trainer/config/baseline_grpo.yaml   # expect 1
   grep -c "ppo_micro_batch_size_per_gpu: 8" verl/trainer/config/user.yaml          # expect 1
   grep -c "project_name: sdpo20icml2027" verl/trainer/config/user.yaml             # expect 1
   ```
   If any check fails, the changes have not been pushed yet; ask the project owner to push them.

2. **Python environment.** The job scripts activate a conda env named `sdql`. Install following README.md → "Option 2: Local Installation", then match the versions the earlier runs used:

   | Package | Version |
   |---|---|
   | python | 3.11.0 |
   | torch | 2.10.0+cu128 |
   | vllm | 0.17.0 |
   | ray | 2.53.0 |
   | transformers | 4.57.1 |
   | flash_attn | 2.8.3 |
   | tensordict | 0.10.0 |
   | torchdata | 0.11.0 |
   | hydra-core | 1.3.2 |
   | accelerate | 1.12.0 |
   | peft | 0.18.0 |
   | numpy | 2.2.6 |
   | wandb | 0.27.2 |

3. **Data.** The `.parquet` files are gitignored, so build them from the committed JSON:
   ```bash
   python data/preprocess.py --data_source datasets/sciknoweval2/all
   ls datasets/sciknoweval2/all/   # must list train.parquet and test.parquet
   ```

4. **Weights & Biases.**
   - Run `wandb login`.
   - Set your entity in `training/verl_training.sh` (line 8 hardcodes `WANDB_ENTITY="harshg5799"`).
   - Also export it for every Ray process. On SLURM, add `export WANDB_ENTITY="<your-entity>"` to `tacc/generalization/ray_wrapper.sh`; Ray workers do not inherit variables set later.
   - Every run logs to the project `sdpo20icml2027`. Do not override `PROJECT_NAME`.

5. **Your machine.** The job scripts in `tacc/jobs/` assume SLURM with 8 nodes × 1 GPU, the repo at `$WORK/SDPO`, and checkpoints under `$WORK` (`HOME_ROOT`).
   - Adapt the account (`$ALLOC` in the commands), partition (`#SBATCH -p`), modules, paths and checkpoint location as needed.
   - Keep 8 GPUs per run: for one 8-GPU node, use `trainer.nnodes=1 trainer.n_gpus_per_node=8`.
   - If you write your own launcher, copy the `ARGS` block from the matching job script and keep the positional arguments from "Commands".
   - Each checkpoint is about 92 GB, so put checkpoints on a filesystem with several TB free.

## Reporting results

- **Metrics:** `val-core/sciknoweval_{biology,chemistry,material,physics}/acc/mean@16` in W&B.
- **For each run,** choose the single validation step with the highest average of the four domains. Report that step's four values and their average.
- **For each algorithm,** pick its single best run and step by that same four-domain average.
- **Never mix steps or runs:** do not combine each domain's best from different steps or different runs.
- **Noise:** reruns of the same config differ by about 2.5 points at their peak. Treat smaller gaps as noise.
- **What to hand back:**
  - one table with columns algorithm, run name, mini-batch, lr, target, λ, α, best step, biology, chemistry, material, physics, average;
  - the W&B link for each run;
  - a short note on any run that crashed, was resumed or needed micro-batch 2.

## Commands

Run everything from the repo root, in a shell where conda is initialised. Every command is complete: the positional arguments after the job script are the run's settings, in the order the job script expects. Copy them exactly. Jobs have a 30-hour limit; a run that hits it continues when you resubmit its line.

### 0. Once per shell

```bash
cd $WORK/SDPO
export ALLOC=<your-slurm-account>
```

### 1. Smoke test

Run one short SDQL job first. It uses its own name, so the real α = 0.5 run will not resume from it.

```bash
TOTAL_TRAINING_STEPS=30 sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True SMOKE-sdql-alpha0.5 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
```

The smoke test passes when the log shows the step-0 validation and then training steps, the step-20 checkpoint saves, and the run appears in W&B in the project `sdpo20icml2027`. Then, with the user's OK, delete its checkpoint folder.

### 2. The 28 runs

Submit each block after the smoke test passes and the user has approved the cost. `sbatch` prints a job ID for each line; record it next to the run name.

GRPO (4 runs):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_grpo.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B FINAL-GRPO-mbs-32-train32-rollout8-lr1e-5-modelQwen/Qwen3-8B
sbatch -A "$ALLOC" tacc/jobs/run_grpo.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B FINAL-GRPO-mbs-8-train32-rollout8-lr1e-5-modelQwen/Qwen3-8B
sbatch -A "$ALLOC" tacc/jobs/run_grpo.slurm datasets/sciknoweval2/all/ 32 8 32 3e-6 Qwen/Qwen3-8B FINAL-GRPO-mbs-32-train32-rollout8-lr3e-6-modelQwen/Qwen3-8B
sbatch -A "$ALLOC" tacc/jobs/run_grpo.slurm datasets/sciknoweval2/all/ 32 8 8 3e-6 Qwen/Qwen3-8B FINAL-GRPO-mbs-8-train32-rollout8-lr3e-6-modelQwen/Qwen3-8B
```

ROVER (2 runs):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_rover.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True 0.01 FINAL-ROVER-mbs-32-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B
sbatch -A "$ALLOC" tacc/jobs/run_rover.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.5 True 0.01 FINAL-ROVER-mbs-8-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B
```

Distil (1 run):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_distil.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 1.0 True 0.01 FINAL-DISTIL-mbs-32-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B 100
```

SDPO (3 runs):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_sdpo.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDPO-noicl-nogae-isclip2.0-cispoclip0.99-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B 100 0.01 False 2.0 False 1.0 False 0.99 none group
sbatch -A "$ALLOC" tacc/jobs/run_sdpo.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDPO-noicl-nogae-isclip2.0-cispoclip0.99-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B 100 0.01 False 2.0 False 1.0 False 0.99 none group
sbatch -A "$ALLOC" tacc/jobs/run_sdpo.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDPO-noicl-nogae-isclip2.0-cispoclip0.99-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B 100 0.01 False 2.0 False 1.0 False 0.99 none group
```

SDQL, mini-batch 32 (9 runs):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 32 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-32-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
```

SDQL, mini-batch 8 (9 runs):

```bash
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.0 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 0.5 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha0.5-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-lambda-lam0.90-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 0.90
sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm datasets/sciknoweval2/all/ 32 8 8 1e-5 Qwen/Qwen3-8B 1.0 True FINAL-SDQL-tqon-policy-lambda-lam1.0-inois-envrwFalse-noclamp-norwbaseline-tur0.01-mbs-8-train32-rollout8-lr1e-5-alpha1.0-modelQwen/Qwen3-8B-topk100 100 True 5.0 -5.0 0.01 on-policy-lambda null False 1.0 False False 1.0 False False 1.0
```

### 3. Resubmitting a run

- **Time limit reached or job cancelled:** run that run's exact line again. It resumes from its latest checkpoint.
- **Out of memory:** run the same line with micro-batch 2 in front. Only for that run:
  ```bash
  MICRO_BATCH_SIZE_PER_GPU=2 sbatch -A "$ALLOC" tacc/jobs/run_sdql.slurm ...   # rest of the line unchanged
  ```

## Don't

- Change any hyperparameter in "Hyperparameters that must stay the same", or any run name or dataset.
- Lower the micro-batch for runs that have not run out of memory.
- Use `tacc/generalization/run_baseline_grpo_all.sh`. It ignores `--dry-run` and submits all four GRPO jobs immediately; use the GRPO commands above instead.
- Turn on reward centering (`USE_REWARD_BASELINE`) or SDPO future returns (`USE_FUTURE_RETURNS`). Neither is part of these experiments.
- Reuse a run name for a different config. Auto-resume would load the wrong checkpoint.
- Commit, push, cancel jobs, edit tracked files or delete files without asking the user.
