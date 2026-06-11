#!/bin/bash
#SBATCH -J vseek_job
#SBATCH -o vseek_job-%j.out
#SBATCH -e vseek_job-%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --mem=0

# USAGE: sbatch run_vseek_job.slurm

echo "Starting job on $(hostname)"
date

# Activate conda environment on head node
source ~/.bashrc
# Fix: Ensure standard CUDA compiler is used, not NVHPC
module load gcc/14.2.0 cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"
conda activate sdql
echo "Conda environment: $CONDA_DEFAULT_ENV"


# Set HuggingFace cache to $WORK directory (persists across jobs, more space)
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"




# -------------------------------------------------------------------------
# Ray Multi-Node Setup & Head Node Detection
# -------------------------------------------------------------------------
# Getting the node names
# Filter out TACC banners - only keep lines that start with alphanumeric (valid hostnames)
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -E '^[a-zA-Z0-9]')
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}

if [ -z "$HEAD_NODE" ]; then
    echo "Error: Could not determine HEAD_NODE from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Raw scontrol output:"
    scontrol show hostnames "$SLURM_JOB_NODELIST" | cat -A
    exit 1
fi

echo "Head Node: $HEAD_NODE"

# Get IP of head node
# TACC systems print banners that can pollute command output
# We try multiple methods to reliably get the IP

# Method 1: Use getent hosts (DNS/hosts lookup, no srun needed)
head_node_ip=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n 1)

# Method 2: If getent failed, try Python socket (very reliable)
if [ -z "$head_node_ip" ]; then
    echo "getent failed, trying Python socket..."
    head_node_ip=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))" 2>/dev/null)
fi

# Method 3: srun hostname -I with grep (original approach)
if [ -z "$head_node_ip" ]; then
    echo "Python failed, trying srun hostname -I..."
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I 2>/dev/null | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -n 1)
fi

if [ -z "$head_node_ip" ]; then
    echo "Error: Could not determine head node IP by any method."
    exit 1
fi

echo "Resolved head node IP: $head_node_ip"

# Port for Ray
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Ray Head IP: $ip_head"

# Clean up any stale Ray processes/sessions from previous runs
echo "Cleaning up stale Ray sessions..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

# Cleanup on workers via SSH
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        ssh "$node_i" "source ~/.bashrc && conda activate vseek-vllm4 && ray stop --force; rm -rf /tmp/ray/session_*" || true
    done
fi

echo "Starting Ray Head on $HEAD_NODE"
export RAY_ADDRESS=$ip_head

# -------------------------------------------------------------------------
# Launch Ray Cluster using SSH (Bypassing srun/ibrun restrictions)
# -------------------------------------------------------------------------

# Ensure wrapper is executable
chmod +x tacc/tests/ray_wrapper.sh

# Capture environment from Head Node (where we know CUDA works)
echo "Capturing environment for workers..."
env | grep -E "^(PATH|LD_LIBRARY_PATH|CUDA_|NCCL_|Ray_|CONDA_)" > worker_env.sh

# 1. Start Ray Head on the Head Node (Local execution)
echo "Launching Ray Head on $HEAD_NODE (local)..."
# We run wrapper locally to setup env, then start ray
tacc/tests/ray_wrapper.sh \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=1 --num-cpus=8 --dashboard-host=0.0.0.0 --block &
sleep 10

# 2. Start Ray Workers on remaining nodes via SSH
if [ $worker_num -gt 0 ]; then
    echo "Launching Ray Workers on $worker_num nodes via SSH..."
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "  -> Starting Worker on $node_i"
        
        # Copy wrapper AND env file to worker
        scp tacc/tests/ray_wrapper.sh "$node_i":~/ray_wrapper.sh
        scp worker_env.sh "$node_i":~/worker_env.sh
        
        # Execute wrapper via SSH
        ssh "$node_i" "chmod +x ~/ray_wrapper.sh && ~/ray_wrapper.sh ray start --address '$ip_head' --num-gpus=1 --num-cpus=8 --block" &
        sleep 5
    done
fi

# Wait for workers to register
echo "Waiting for cluster to initialize..."
for i in {1..30}; do
    if tacc/tests/ray_wrapper.sh ray status > /dev/null 2>&1; then
        echo "Ray cluster is UP!"
        break
    fi
    echo "Waiting for Ray head to be ready... ($i/30)"
    sleep 5
done

# Verify status again with output
tacc/tests/ray_wrapper.sh ray status || { echo "Ray failed to start"; exit 1; }
sleep 5
echo "Ray cluster status:"
# Run status check inside wrapper to ensure environment is right
tacc/tests/ray_wrapper.sh ray status || echo "Ray status check failed (might be too early)"

# Verify we have enough GPUs
total_gpus=$(tacc/tests/ray_wrapper.sh ray status | grep -oP '\d+\.\d+(?=/\d+\.\d+ GPU)' | tail -1 || echo "0")
echo "Total GPUs in cluster: $total_gpus"

# -------------------------------------------------------------------------

# Run the training job
echo "Starting training..."

# IMPORTANT: The training job typically needs access to GPUs. 
# We unset CUDA_VISIBLE_DEVICES so the training script can see all available GPUs (including GPU 0).
# Note: If the server is using GPU 0 heavily, this might cause contention.


# Setup training variables
# Use SLURM_SUBMIT_DIR or hardcoded path (BASH_SOURCE doesn't work reliably in SLURM)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${WORK}/SDPO}"
    # If submitted from scripts/tacc, go up two levels
    if [[ "$PROJECT_ROOT" == */SDPO ]]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
fi
# Fallback to hardcoded path if SLURM_SUBMIT_DIR is wrong (e.g., /var/spool)
if [[ ! -d "$PROJECT_ROOT/src/SDPO" ]]; then
    echo "Warning: PROJECT_ROOT=$PROJECT_ROOT doesn't contain src/vseek, using hardcoded path"
    PROJECT_ROOT="${WORK}/SDPO"
fi


# Run the training job
echo "Starting training..."


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
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144} # reduce from 6144 to avoid actor OOM
TEMPLATE_LENGTH=512  # heuristic upper bound, not enforced
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
HOME_ROOT=${WORK}
# Allow overriding experiment name suffix
SUFFIX=${1:-"local_grpo_sciknoweval_all"}

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
