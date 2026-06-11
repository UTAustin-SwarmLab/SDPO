#!/bin/bash
#SBATCH -J sdql_test
#SBATCH -o sdql_test-%j.out
#SBATCH -e sdql_test-%j.err
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH --mem=0

# Usage: sbatch tacc/tests/run_sdql.slurm
# Purpose: lightweight SDQL smoke test on TACC

set -euo pipefail

echo "Starting SDQL smoke test on $(hostname)"
date

source ~/.bashrc
module load gcc/14.2.0 cuda/12.8
export CC
CC=$(which gcc)
export CXX
CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"
conda activate sdql
echo "Conda environment: $CONDA_DEFAULT_ENV"

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -E '^[a-zA-Z0-9]')
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]:-}
if [[ -z "$HEAD_NODE" ]]; then
    echo "Error: Could not determine HEAD_NODE from: $SLURM_JOB_NODELIST"
    exit 1
fi

head_node_ip=$(getent hosts "$HEAD_NODE" | awk '{print $1; exit}')
if [[ -z "$head_node_ip" ]]; then
    head_node_ip=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))" 2>/dev/null || true)
fi
if [[ -z "$head_node_ip" ]]; then
    echo "Error: Could not resolve head node IP for $HEAD_NODE"
    exit 1
fi

port=${RAY_PORT:-6379}
ip_head="${head_node_ip}:${port}"
export RAY_ADDRESS="$ip_head"
echo "Ray head: $ip_head"

echo "Cleaning up stale Ray sessions..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    ssh "$node_i" "source ~/.bashrc && conda activate sdql && ray stop --force; rm -rf /tmp/ray/session_*" || true
done

WRAPPER_SCRIPT="tacc/tests/ray_wrapper.sh"
if [[ ! -f "$WRAPPER_SCRIPT" ]]; then
    echo "Error: Missing wrapper at $WRAPPER_SCRIPT"
    exit 1
fi
chmod +x "$WRAPPER_SCRIPT"

env | grep -E "^(PATH|LD_LIBRARY_PATH|CUDA_|NCCL_|CONDA_)" > worker_env.sh

echo "Launching Ray head on $HEAD_NODE"
"$WRAPPER_SCRIPT" ray start --head --node-ip-address="$head_node_ip" --port="$port" --num-gpus=1 --num-cpus=8 --block &
sleep 10

if [[ $worker_num -gt 0 ]]; then
    echo "Launching $worker_num Ray worker(s)"
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        scp "$WRAPPER_SCRIPT" "$node_i":~/ray_wrapper.sh
        scp worker_env.sh "$node_i":~/worker_env.sh
        ssh "$node_i" "chmod +x ~/ray_wrapper.sh && ~/ray_wrapper.sh ray start --address '$ip_head' --num-gpus=1 --num-cpus=8 --block" &
        sleep 4
    done
fi

echo "Waiting for Ray cluster readiness..."
for i in {1..24}; do
    if "$WRAPPER_SCRIPT" ray status >/dev/null 2>&1; then
        echo "Ray cluster is ready."
        break
    fi
    echo "  waiting ($i/24)"
    sleep 5
done
"$WRAPPER_SCRIPT" ray status || { echo "Ray failed to start"; exit 1; }

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${WORK}/SDPO}"
if [[ ! -d "$PROJECT_ROOT/verl" ]]; then
    PROJECT_ROOT="${WORK}/SDPO"
fi
if [[ ! -d "$PROJECT_ROOT/verl" ]]; then
    echo "Error: could not locate project root with verl/ directory"
    exit 1
fi

CONFIG_NAME="sdql"
DATA_PATH="${DATA_PATH:-datasets/sciknoweval/all}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-0.5}"
DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-True}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TEMPLATE_LENGTH=512
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export TASK="${TASK:-$DATA_PATH}"
OUTPUT_DIR="${WORK}/output/SDQL-test"
mkdir -p "$OUTPUT_DIR"

EXP_NAME="${EXP_NAME:-SMOKE-SDQL-$(date +%Y%m%d-%H%M%S)}"
ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
trainer.experiment_name=$EXP_NAME \
vars.task=$TASK \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.dir=$PROJECT_ROOT \
vars.log_dir=$OUTPUT_DIR \
trainer.group_name=SDQL-test \
trainer.nnodes=$SLURM_JOB_NUM_NODES \
trainer.n_gpus_per_node=1 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.distillation_topk=50 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.val_kwargs.n=2 \
trainer.test_freq=5 \
actor_rollout_ref.rollout.agent.num_workers=2 \
actor_rollout_ref.rollout.max_num_seqs=32 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "Starting SDQL smoke-test training run..."
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
