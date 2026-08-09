#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
DATA_PATH="datasets/sciknoweval/all"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=4
MINI_BATCH_SIZE=16
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
USE_FUTURE_RETURNS=${USE_FUTURE_RETURNS:-False}
USE_FUTURE_RETURNS_BASELINE=${USE_FUTURE_RETURNS_BASELINE:-False}
GAMMA=${GAMMA:-1.0}
MODEL_PATH="Qwen/Qwen3-4B"
RAY_DASHBOARD_PORT=8170
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=6144
TEMPLATE_LENGTH=512  # heuristic upper bound, not enforced
MAX_FEEDBACK_LENGTH=$((MAX_RESPONSE_LENGTH + 512))
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_FEEDBACK_LENGTH + MAX_RESPONSE_LENGTH))
HOME_ROOT=/home/hg22723/projects/SDPO
# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdpo_sciknoweval_all"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME_ROOT}/output/SDPO-local"
mkdir -p "$LOCAL_OUTPUT_DIR"

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.dir=$PROJECT_ROOT \
vars.log_dir=$LOCAL_OUTPUT_DIR \
trainer.group_name=SDPO-local \
trainer.nnodes=1 \
trainer.n_gpus_per_node=4 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.use_future_returns=$USE_FUTURE_RETURNS \
actor_rollout_ref.actor.self_distillation.gamma=$GAMMA \
actor_rollout_ref.actor.self_distillation.use_future_returns_baseline=$USE_FUTURE_RETURNS_BASELINE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
trainer.test_freq=20 \
actor_rollout_ref.rollout.agent.num_workers=4 \
actor_rollout_ref.rollout.max_num_seqs=64 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"\

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "use_future_returns: $USE_FUTURE_RETURNS (gamma=$GAMMA)"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
