#!/bin/bash

# Usage: ./run_sdql_all.sh [--dry-run]
#
# TACC generalization sweep for SDQL baseline.
# Mirrors experiments/generalization/run_sdql_all.sh.

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TACC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATHS=(
    "datasets/sciknoweval2/all/"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(16)
LRS=(3e-6)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
ALPHAS=(0.0 0.5 1.0)
FULL_LOGIT_DISTILLATION=True
CLAMP_HIGH=5.0
CLAMP_LOW=-5.0
TOPK=50
ALPHAS=(0.5)
TEACHER_UPDATE_RATE=0.05

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                            for DATA_PATH in "${DATA_PATHS[@]}"; do
                                EXP_NAME="FINAL-SDQL-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-model${MODEL_PATH}-topk${TOPK}"
                                CMD=(sbatch -A ASC26054 "$TACC_DIR/jobs/run_sdql.slurm" "${DATA_PATH}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_BATCH_SIZE}" "${MINI_BATCH_SIZE}" "${LR}" "${MODEL_PATH}" "${ALPHA}" "${DONTS_REPROMPT_ON_SELF_SUCCESS}" "${EXP_NAME}" "${TOPK}" "${FULL_LOGIT_DISTILLATION}" "${CLAMP_HIGH}" "${CLAMP_LOW}" "${TEACHER_UPDATE_RATE}")
                                if [[ "$DRY_RUN" == true ]]; then
                                    printf '%q ' "${CMD[@]}"
                                    echo
                                else
                                    "${CMD[@]}"
                                fi
                                sleep 90
                            done
                        done
                    done
                done
            done
        done
    done
done
