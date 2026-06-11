#!/bin/bash
#SBATCH -J grpo_job
#SBATCH -o grpo_job-%j.out
#SBATCH -e grpo_job-%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1    # Important for Ray: 1 task (daemon) per node
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH --mem=0

# USAGE: sbatch run_grpo_all_jobs.slurm [em|puls]

echo "Starting job on $(hostname)"
date

# ==========================================
# 1. CONFIGURATION & PORTS
# ==========================================
# You can change these ports explicitly here
export MASTER_PORT=${MASTER_PORT:-29651}
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1
export RAY_memory_usage_threshold=0.95

export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8265
export SERVER_PORT=9000
export RAY_MIN_WORKER_PORT=10002
export RAY_MAX_WORKER_PORT=19999

# Environment Setup
source ~/.bashrc
module load gcc/14.2.0 cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"

conda activate sdql
echo "Conda environment: $CONDA_DEFAULT_ENV"

# HuggingFace Cache
export HF_HOME="${SCRATCH}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# Setup Paths
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${WORK}/SDPO}"
MODEL_PATH="Qwen/Qwen3-8B"
PROJECT_NAME="SDPO"


WRAPPER_SCRIPT="${WORK}/SDPO/tacc/generalization/ray_wrapper.sh"
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found at $WRAPPER_SCRIPT"
    exit 1
fi
chmod +x "$WRAPPER_SCRIPT"

# ==========================================
# 2. HEAD NODE IDENTIFICATION
# ==========================================
# Get the first node from the nodelist to be the head
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}

if [ -z "$HEAD_NODE" ]; then
    echo "Error: Could not determine HEAD_NODE from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Raw scontrol output:"
    scontrol show hostnames "$SLURM_JOB_NODELIST" | cat -A
    exit 1
fi

echo "Head Node: $HEAD_NODE"

# Get IP address (robust method)
# On TACC/HPC, hostname -I often returns multiple IPs (InfiniBand, Ethernet, Docker)
# We usually want the one that starts with 10. (internal) or 192. or 172.
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')

if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))")
    
fi

echo "Head Node: $HEAD_NODE"
echo "Head IP:   $HEAD_IP"

# Export for Ray
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

# Clean up previous Ray sessions locally
echo "Cleaning up stale Ray sessions..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

# ==========================================
# 3. LAUNCH RAY CLUSTER
# ==========================================

echo "----------------------------------------"
echo "Launching Ray Cluster"
echo "----------------------------------------"

# A. Start Ray Head (Local execution on the first node)
# We do not use srun here because the batch script is already running on the head node.
echo "Starting Ray Head on $HEAD_NODE..."
numactl --cpunodebind=0 --membind=0 --physcpubind=0-71" $WRAPPER_SCRIPT" ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=$RAY_PORT \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --min-worker-port=$RAY_MIN_WORKER_PORT \
    --max-worker-port=$RAY_MAX_WORKER_PORT \
    --num-gpus=1 \
    --block &

# Store PID to kill later if needed, though 'ray stop' handles it
HEAD_PID=$!
sleep 10

# B. Start Ray Workers (Remote execution via srun)
# --relative=1 means "start from the 2nd node assigned to this job"
# This prevents srun from trying to start a worker on the Head Node
NUM_WORKERS=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= NUM_WORKERS; i++)); do
    node_i=${nodes_array[i]}
    echo "Starting Ray Worker on $node_i"
    # We use srun to execute the start command on worker nodes.
    # Note: We explicitly source bashrc/conda again to ensure the non-interactive srun shell has the env.
    srun --nodes=1 --ntasks=1 -w $node_i \
        bash -c "source ~/.bashrc && conda activate sdql && \
                numactl --cpunodebind=0 --membind=0 --physcpubind=0-71 $WRAPPER_SCRIPT ray start --address $HEAD_IP:$RAY_PORT --num-gpus=1 --num-cpus=32 --block" &
    sleep 5
done

# C. Wait for Cluster Initialization
echo "Waiting for Ray cluster to initialize..."
for i in {1..30}; do
    # Check if we have the expected number of nodes
    alive_nodes=$(ray status --address "$HEAD_IP:$RAY_PORT" 2>/dev/null | grep "node_" | wc -l)
    if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
        echo "Ray Cluster Ready! ($alive_nodes nodes connected)"
        break
    fi
    sleep 5
done

# Print Status
ray status --address "$HEAD_IP:$RAY_PORT"


#!/bin/bash

# Usage: ./run_local_grpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker, use everything else
export NCCL_IB_DISABLE=0              # Enable InfiniBand/RDMA
export NCCL_NET_GDR_LEVEL=5           # Enable GPU Direct RDMA
export NCCL_DEBUG=WARN
module load gcc/14.2.0 cuda/12.8
CONFIG_NAME="baseline_grpo"

# Default to ToolUse dataset
# Arguments:  DATA_PATH, TRAIN_BATCH_SIZE, ROLLOUT_BATCH_SIZE, MINI_BATCH_SIZE, LR, MODEL_PATH, EXPERIMENT_NAME
DATA_PATH=${1:-"datasets/sciknoweval/all"}

# Hyperparameters (OOM-safe defaults; override via env if needed)
TRAIN_BATCH_SIZE=${2:-32}
ROLLOUT_BATCH_SIZE=${3:-8}
MINI_BATCH_SIZE=${4:-8}
LR=${5:-1e-5}
MODEL_PATH=${6:-"Qwen/Qwen3-4B"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8169}
HOME_ROOT=${WORK}
# Allow overriding experiment name suffix
SUFFIX=${7:-"local_grpo_physics"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="${WORK}/SDPO"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME_ROOT}/output/GRPO-local"
# Suggested by PyTorch OOM hint to reduce fragmentation-related failures.
mkdir -p "$LOCAL_OUTPUT_DIR"

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}
export TASK="${TASK:-$DATA_PATH}"
echo "Data: $DATA_PATH"
echo "Task: $TASK"
# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
vars.dir=$PROJECT_ROOT \
trainer.experiment_name=$EXP_NAME \
vars.task=$TASK \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.log_dir=$LOCAL_OUTPUT_DIR \
trainer.group_name=GRPO-local \
trainer.nnodes=8 \
trainer.n_gpus_per_node=1 \
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
