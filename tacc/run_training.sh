#!/bin/bash
set -euo pipefail

# Compute-node entrypoint invoked from generated Slurm job scripts.

TACC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tacc/config.sh
source "${TACC_DIR}/config.sh"
# shellcheck source=tacc/ray_cluster.sh
source "${TACC_DIR}/ray_cluster.sh"

: "${EXP_NAME:?EXP_NAME must be set}"
: "${CONFIG_NAME:?CONFIG_NAME must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"
: "${SCRIPT_ARGS:?SCRIPT_ARGS must be set}"
: "${OUTPUT_SUBDIR:?OUTPUT_SUBDIR must be set}"

setup_env() {
    # Customize for your TACC environment. Examples:
    #   module load python3/3.12.0
    #   source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate sdpo
    #   module load tacc-apptainer && apptainer exec --nv "$SDPO_CONTAINER" bash -lc "..."

    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export USER="${USER:-$(whoami)}"
    cd "$PROJECT_ROOT"
}

setup_env
start_ray_cluster_if_needed

ICL_VALIDATION="${ICL_VALIDATION:-True}"
ICL_VALIDATION_POLICY="${ICL_VALIDATION_POLICY:-student}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"

# SAVE_FREQ defaults to test_freq when unset so checkpoints land after validation.
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-${TEST_FREQ}}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"

COMMON_ARGS="vars.dir=${PROJECT_ROOT} \
vars.log_dir=${LOG_DIR} \
vars.ckpt_dir=${CKPT_DIR} \
trainer.nnodes=${NODES} \
trainer.n_gpus_per_node=${GPUS_PER_NODE} \
trainer.icl_validation.enable=${ICL_VALIDATION} \
trainer.icl_validation.policy=${ICL_VALIDATION_POLICY} \
trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
trainer.test_freq=${TEST_FREQ} \
trainer.save_freq=${SAVE_FREQ} \
trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"

RUN_CMD="bash ${PROJECT_ROOT}/training/verl_training.sh ${EXP_NAME} ${CONFIG_NAME} ${DATA_PATH} ${SCRIPT_ARGS} ${COMMON_ARGS}"

echo "Running on $(hostname)"
echo "Experiment: ${EXP_NAME}"
echo "Config: ${CONFIG_NAME}"
echo "Data: ${DATA_PATH}"
echo "Nodes: ${NODES}, GPUs per node: ${GPUS_PER_NODE}"

run_training_on_ray_head "$RUN_CMD"
