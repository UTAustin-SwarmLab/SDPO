#!/bin/bash

# Usage: ./run_baseline_grpo_all.sh [--dry-run]
#
# TACC generalization sweep for GRPO baseline.
# Mirrors experiments/generalization/run_baseline_grpo_all.sh.

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TACC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=tacc/submit.sh

CONFIG_NAME="baseline_grpo"
OUTPUT_SUBDIR="GRPO"

DATA_PATHS=(
    "datasets/sciknoweval/all/"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(16)
LRS=(3e-6)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                        EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}"
                        sbatch -A ASC26054 "$TACC_DIR/jobs/run_grpo.slurm" "${DATA_PATH}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_BATCH_SIZE}" "${MINI_BATCH_SIZE}" "${LR}" "${MODEL_PATH}" "${EXP_NAME}"
                    done
                done
            done
        done
    done
done
