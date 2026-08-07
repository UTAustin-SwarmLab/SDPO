#!/bin/bash

# Usage: ENABLE_ICL=True ./run_sdpo_nofulllogit_all.sh [--dry-run]
#
# TACC generalization sweep for SDPO with sampled-token (non-full-logit) distillation.
# Mirrors tacc/generalization/run_sdpo_all.sh.
# Set ENABLE_ICL=True to use sdpo_icl config (default: False → sdpo).

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

ENABLE_ICL="${ENABLE_ICL:-False}"
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(16)
LRS=(3e-6)
ALPHAS=(0.0 0.5 1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
# Sampled-token path does not use top-k logits.
TOPK=null
TEACHER_UPDATE_RATE=0.05
CLAMP_HIGH=2.0
CLAMP_LOW=-2.0
USE_REWARD_CLAMP="${USE_REWARD_CLAMP:-True}"
# PPO-style symmetric IS clip for sampled-token path: ratio ∈ [1 - is_clip, 1 + is_clip]
IS_CLIPS=(0.2)
MODEL_PATHS=(
    "allenai/Olmo-3-7B-Instruct"
    "Qwen/Qwen3-8B"
)

if [[ "${ENABLE_ICL}" == "True" || "${ENABLE_ICL}" == "true" || "${ENABLE_ICL}" == "1" ]]; then
    ICL_TAG="icl"
else
    ICL_TAG="noicl"
fi

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
                            for IS_CLIP in "${IS_CLIPS[@]}"; do
                                for DATA_PATH in "${DATA_PATHS[@]}"; do
                                    EXP_NAME="FINAL-SDPO-nofulllogit-${ICL_TAG}-${CLAMP_TAG}-isclip${IS_CLIP}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-model${MODEL_PATH}"
                                    CMD=(sbatch -A ASC26054 "$TACC_DIR/jobs/run_sdpo_nofulllogit.slurm" "${DATA_PATH}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_BATCH_SIZE}" "${MINI_BATCH_SIZE}" "${LR}" "${MODEL_PATH}" "${ALPHA}" "${DONTS_REPROMPT_ON_SELF_SUCCESS}" "${EXP_NAME}" "${TOPK}" "${TEACHER_UPDATE_RATE}" "${ENABLE_ICL}" "${CLAMP_HIGH}" "${CLAMP_LOW}" "${USE_REWARD_CLAMP}" "${IS_CLIP}")
                                    if [[ "$DRY_RUN" == true ]]; then
                                        printf '%q ' "${CMD[@]}"
                                        echo
                                    else
                                        "${CMD[@]}"
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
