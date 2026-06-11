#!/bin/bash
#SBATCH -J sdpo_job
#SBATCH -o sdpo_job-%j.out
#SBATCH -e sdpo_job-%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --mem=0

# USAGE: sbatch tacc/tests/run_sdpo.slurm

echo "Starting job on $(hostname)"
date

# Activate conda environment on head node
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

# Set HuggingFace cache to $WORK directory (persists across jobs, more space)
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# -------------------------------------------------------------------------
# Ray Multi-Node Setup & Head Node Detection
# -------------------------------------------------------------------------
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
head_node_ip=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n 1)
if [ -z "$head_node_ip" ]; then
    echo "getent failed, trying Python socket..."
    head_node_ip=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))" 2>/dev/null)
fi
if [ -z "$head_node_ip" ]; then
    echo "Python failed, trying srun hostname -I..."
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I 2>/dev/null | grep -oE '[0-9]{1,3}(\.[0-9]{1,3}){3}' | head -n 1)
fi
if [ -z "$head_node_ip" ]; then
    echo "Error: Could not determine head node IP by any method."
    exit 1
fi

echo "Resolved head node IP: $head_node_ip"

port=6379
ip_head=$head_node_ip:$port
export ip_head
export RAY_ADDRESS=$ip_head
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
        ssh "$node_i" "source ~/.bashrc && conda activate sdql && ray stop --force; rm -rf /tmp/ray/session_*" || true
    done
fi

# -------------------------------------------------------------------------
# Launch Ray Cluster using SSH
# -------------------------------------------------------------------------
chmod +x tacc/tests/ray_wrapper.sh
echo "Capturing environment for workers..."
env | grep -E "^(PATH|LD_LIBRARY_PATH|CUDA_|NCCL_|Ray_|CONDA_)" > worker_env.sh

echo "Launching Ray Head on $HEAD_NODE (local)..."
tacc/tests/ray_wrapper.sh \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=1 --num-cpus=8 --dashboard-host=0.0.0.0 --block &
sleep 10

if [ $worker_num -gt 0 ]; then
    echo "Launching Ray Workers on $worker_num nodes via SSH..."
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "  -> Starting Worker on $node_i"
        scp tacc/tests/ray_wrapper.sh "$node_i":~/ray_wrapper.sh
        scp worker_env.sh "$node_i":~/worker_env.sh
        ssh "$node_i" "chmod +x ~/ray_wrapper.sh && ~/ray_wrapper.sh ray start --address '$ip_head' --num-gpus=1 --num-cpus=8 --block" &
        sleep 5
    done
fi

echo "Waiting for cluster to initialize..."
for i in {1..30}; do
    if tacc/tests/ray_wrapper.sh ray status > /dev/null 2>&1; then
        echo "Ray cluster is UP!"
        break
    fi
    echo "Waiting for Ray head to be ready... ($i/30)"
    sleep 5
done
tacc/tests/ray_wrapper.sh ray status || { echo "Ray failed to start"; exit 1; }

# -------------------------------------------------------------------------
# SDPO quick-test training config (from run_local_sdpo.sh)
# -------------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${WORK}/SDPO}"
if [[ ! -d "$PROJECT_ROOT/verl" ]]; then
    PROJECT_ROOT="${WORK}/SDPO"
fi

CONFIG_NAME="sdpo"
DATA_PATH="${DATA_PATH:-datasets/sciknoweval/all}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
LR=${LR:-1e-5}
ALPHA=${ALPHA:-0.5}
DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS:-True}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8170}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144}
TEMPLATE_LENGTH=512
MAX_FEEDBACK_LENGTH=$((MAX_RESPONSE_LENGTH + 512))
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_FEEDBACK_LENGTH + MAX_RESPONSE_LENGTH))
HOME_ROOT=${WORK}
SUFFIX=${1:-"local_sdpo_sciknoweval_all"}

export PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME_ROOT}/output/SDPO-local"
mkdir -p "$LOCAL_OUTPUT_DIR"

export USER=${USER:-$(whoami)}
export TASK="${TASK:-$DATA_PATH}"

EXP_NAME="LOCAL-SDPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
trainer.experiment_name=$EXP_NAME \
vars.task=$TASK \
max_model_len=$MAX_MODEL_LEN \
vars.dir=$PROJECT_ROOT \
vars.log_dir=$LOCAL_OUTPUT_DIR \
trainer.group_name=SDPO-local \
trainer.nnodes=8 \
trainer.n_gpus_per_node=1 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
trainer.test_freq=20 \
actor_rollout_ref.rollout.agent.num_workers=4 \
actor_rollout_ref.rollout.max_num_seqs=64 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Task: $TASK"
echo "Model: $MODEL_PATH"
echo "Ray dashboard port: $RAY_DASHBOARD_PORT"
echo "GPUs per node: 1"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
