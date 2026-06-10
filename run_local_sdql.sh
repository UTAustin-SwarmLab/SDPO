#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdql"

# Default to ToolUse dataset
DATA_PATH="datasets/new/gsm8k"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=16
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
RAY_DASHBOARD_PORT=8171

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdql"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME}/output/SDQL-local"
mkdir -p "$LOCAL_OUTPUT_DIR"

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-SDQL-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
vars.dir=$PROJECT_ROOT \
vars.log_dir=$LOCAL_OUTPUT_DIR \
trainer.group_name=SDQL-local \
trainer.nnodes=1 \
trainer.n_gpus_per_node=2 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=8 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"\


echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
