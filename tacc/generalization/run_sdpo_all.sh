#!/bin/bash

# Usage: USE_FUTURE_RETURNS=True USE_FUTURE_RETURNS_BASELINE=True FUTURE_RETURNS_BASELINE_MODE=group FUTURE_RETURNS_REWARD_SCALE=none GAMMA=1.0 ENABLE_ICL=True ./run_sdpo_all.sh [--dry-run]
#
# TACC generalization sweep for SDPO baseline.
# Mirrors experiments/generalization/run_sdpo_all.sh.
# Set ENABLE_ICL=True to use sdpo_icl config (default: False → sdpo).
# Set USE_FUTURE_RETURNS=True to enable discounted future-KL returns (default: False).
# Set USE_FUTURE_RETURNS_BASELINE=True to subtract leave-one-out baseline (default: False;
# only applied when USE_FUTURE_RETURNS is enabled; appends -baseline-<mode> to the gae tag).
# Set FUTURE_RETURNS_BASELINE_MODE=group|batch (default: group). group = per-prompt LOO;
# batch = full-microbatch LOO.
# Set FUTURE_RETURNS_REWARD_SCALE=none|remaining|seq_len (default: none). Divides r_t
# before baseline/cumsum; remaining = N_rem(t), seq_len = T_i. Appends -s* to the gae tag.

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
USE_FUTURE_RETURNS="${USE_FUTURE_RETURNS:-False}"
USE_FUTURE_RETURNS_BASELINE="${USE_FUTURE_RETURNS_BASELINE:-False}"
FUTURE_RETURNS_BASELINE_MODE="${FUTURE_RETURNS_BASELINE_MODE:-group}"
FUTURE_RETURNS_REWARD_SCALE="${FUTURE_RETURNS_REWARD_SCALE:-none}"
GAMMA="${GAMMA:-1.0}"
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(16)
LRS=(1e-5)
ALPHAS=(0.0 0.5 1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
TOPK=100
TEACHER_UPDATE_RATE=0.01
# One-sided IS ratio clip for full-logit distill KL (null disables IS weighting).
IS_CLIPS=(2.0)
# CISPO-style symmetric clip for the future-returns PG term: ratio ∈ [1-cispo_clip, 1+cispo_clip].
CISPO_CLIPS=(0.99)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

if [[ "${ENABLE_ICL}" == "True" || "${ENABLE_ICL}" == "true" || "${ENABLE_ICL}" == "1" ]]; then
    ICL_TAG="icl"
else
    ICL_TAG="noicl"
fi

if [[ "${USE_FUTURE_RETURNS}" == "True" || "${USE_FUTURE_RETURNS}" == "true" || "${USE_FUTURE_RETURNS}" == "1" ]]; then
    RETURNS_TAG="gae${GAMMA}"
    if [[ "${USE_FUTURE_RETURNS_BASELINE}" == "True" || "${USE_FUTURE_RETURNS_BASELINE}" == "true" || "${USE_FUTURE_RETURNS_BASELINE}" == "1" ]]; then
        RETURNS_TAG="${RETURNS_TAG}-baseline-${FUTURE_RETURNS_BASELINE_MODE}"
    fi
    RETURNS_TAG="${RETURNS_TAG}-rs${FUTURE_RETURNS_REWARD_SCALE}"
else
    RETURNS_TAG="nogae"
fi

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                            for IS_CLIP in "${IS_CLIPS[@]}"; do
                                for CISPO_CLIP in "${CISPO_CLIPS[@]}"; do
                                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                                        EXP_NAME="FINAL-SDPO-${ICL_TAG}-${RETURNS_TAG}-isclip${IS_CLIP}-cispoclip${CISPO_CLIP}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-model${MODEL_PATH}"
                                        CMD=(sbatch -A ASC26054 "$TACC_DIR/jobs/run_sdpo.slurm" "${DATA_PATH}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_BATCH_SIZE}" "${MINI_BATCH_SIZE}" "${LR}" "${MODEL_PATH}" "${ALPHA}" "${DONTS_REPROMPT_ON_SELF_SUCCESS}" "${EXP_NAME}" "${TOPK}" "${TEACHER_UPDATE_RATE}" "${ENABLE_ICL}" "${IS_CLIP}" "${USE_FUTURE_RETURNS}" "${GAMMA}" "${USE_FUTURE_RETURNS_BASELINE}" "${CISPO_CLIP}" "${FUTURE_RETURNS_REWARD_SCALE}" "${FUTURE_RETURNS_BASELINE_MODE}")
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
done
