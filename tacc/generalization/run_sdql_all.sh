#!/bin/bash

# Usage: ./run_sdql_all.sh [--dry-run]
#
# TACC generalization sweep for SDQL baseline.
# Mirrors experiments/generalization/run_sdql_all.sh.
#
# SDQL-specific / shared distillation flags (override via env):
#   TARGET_Q_MODES, IS_CLIPS, USE_ENV_REWARDS, ENV_REWARD_SCALE,
#   USE_REWARD_CLAMP, DISTILLATION_ADD_TAIL, GAMMA, LAMBDAS, INCLUDE_ENVIRONMENT_FEEDBACK,
#   USE_REWARD_BASELINES (True False)

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
MINI_BATCH_SIZES=(32)
LRS=(3e-6)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
ALPHAS=(0.0 0.5 1.0)
FULL_LOGIT_DISTILLATION=True
CLAMP_HIGH=5.0
CLAMP_LOW=-5.0
TOPK=100
TEACHER_UPDATE_RATE=0.05

# Shared / SDQL loss flags (aligned with sdql.yaml + other TACC trainers)
TARGET_Q_MODES=(uniform on-policy)
# null disables loss-level IS weighting (rover-style).
IS_CLIPS=(null)
USE_ENV_REWARDS=(False)
ENV_REWARD_SCALE=1.0
USE_REWARD_CLAMP=False
DISTILLATION_ADD_TAIL=False
GAMMA=1.0
LAMBDAS=(0.95)
INCLUDE_ENVIRONMENT_FEEDBACK=False
# Sweep both leave-one-out baseline on and off.
USE_REWARD_BASELINES=(True False)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

if [[ "${USE_REWARD_CLAMP}" == "True" || "${USE_REWARD_CLAMP}" == "true" || "${USE_REWARD_CLAMP}" == "1" ]]; then
    CLAMP_TAG="clamp${CLAMP_LOW}_${CLAMP_HIGH}"
else
    CLAMP_TAG="noclamp"
fi

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                            for TARGET_Q_MODE in "${TARGET_Q_MODES[@]}"; do
                                for IS_CLIP in "${IS_CLIPS[@]}"; do
                                    for USE_ENV_REWARD in "${USE_ENV_REWARDS[@]}"; do
                                        for LAMBDA_ in "${LAMBDAS[@]}"; do
                                            for USE_REWARD_BASELINE in "${USE_REWARD_BASELINES[@]}"; do
                                                for DATA_PATH in "${DATA_PATHS[@]}"; do
                                                if [[ "${IS_CLIP}" == "null" || -z "${IS_CLIP}" ]]; then
                                                    IS_TAG="inois"
                                                else
                                                    IS_TAG="isclip${IS_CLIP}"
                                                fi
                                                if [[ "${USE_REWARD_BASELINE}" == "True" || "${USE_REWARD_BASELINE}" == "true" || "${USE_REWARD_BASELINE}" == "1" ]]; then
                                                    BASELINE_TAG="rwbaseline"
                                                else
                                                    BASELINE_TAG="norwbaseline"
                                                fi
                                                EXP_NAME="FINAL-SDQL-tq${TARGET_Q_MODE}-lam${LAMBDA_}-${IS_TAG}-envrw${USE_ENV_REWARD}-${CLAMP_TAG}-${BASELINE_TAG}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-model${MODEL_PATH}-topk${TOPK}"
                                                CMD=(
                                                    sbatch
                                                    -A ASC26054
                                                    "$TACC_DIR/jobs/run_sdql.slurm"
                                                    "${DATA_PATH}"
                                                    "${TRAIN_BATCH_SIZE}"
                                                    "${ROLLOUT_BATCH_SIZE}"
                                                    "${MINI_BATCH_SIZE}"
                                                    "${LR}"
                                                    "${MODEL_PATH}"
                                                    "${ALPHA}"
                                                    "${DONTS_REPROMPT_ON_SELF_SUCCESS}"
                                                    "${EXP_NAME}"
                                                    "${TOPK}"
                                                    "${FULL_LOGIT_DISTILLATION}"
                                                    "${CLAMP_HIGH}"
                                                    "${CLAMP_LOW}"
                                                    "${TEACHER_UPDATE_RATE}"
                                                    "${TARGET_Q_MODE}"
                                                    "${IS_CLIP}"
                                                    "${USE_ENV_REWARD}"
                                                    "${ENV_REWARD_SCALE}"
                                                    "${USE_REWARD_CLAMP}"
                                                    "${DISTILLATION_ADD_TAIL}"
                                                    "${GAMMA}"
                                                    "${INCLUDE_ENVIRONMENT_FEEDBACK}"
                                                    "${USE_REWARD_BASELINE}"
                                                    "${LAMBDA_}"
                                                )
                                                if [[ "$DRY_RUN" == true ]]; then
                                                    printf '%q ' "${CMD[@]}"
                                                    echo
                                                else
                                                    "${CMD[@]}"
                                                    sleep 90
                                                fi
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
