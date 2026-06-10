#!/bin/bash

# Usage: ./run_local_grpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="baseline_grpo"

# Default to ToolUse dataset
DATA_PATH="datasets/sciknoweval/all"

# Hyperparameters (OOM-safe defaults; override via env if needed)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
LR=${LR:-1e-5}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8169}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048} # gsm8k 1024, physics 2048
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096} # reduce from 6144 to avoid actor OOM
TEMPLATE_LENGTH=512  # heuristic upper bound, not enforced
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
HOME_ROOT=/home/hg22723/projects/SDPO
# Allow overriding experiment name suffix
SUFFIX=${1:-"local_grpo_physics"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME_ROOT}/output/GRPO-local"
# Suggested by PyTorch OOM hint to reduce fragmentation-related failures.
mkdir -p "$LOCAL_OUTPUT_DIR"

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
vars.dir=$PROJECT_ROOT \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.log_dir=$LOCAL_OUTPUT_DIR \
trainer.group_name=GRPO-local \
trainer.nnodes=1 \
trainer.n_gpus_per_node=2 \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
trainer.test_freq=20 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
# +ray_kwargs.ray_init.dashboard_port=$RAY_DASHBOARD_PORT"


echo "----------------------------------------------------------------"
echo "Starting Local GRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Ray dashboard port: $RAY_DASHBOARD_PORT"
echo "GPUs per node: 1"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
