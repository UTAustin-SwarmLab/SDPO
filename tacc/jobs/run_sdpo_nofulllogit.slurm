#!/bin/bash
#SBATCH -J sdpo_nofulllogit
#SBATCH -o sdpo_nofulllogit-%j.out
#SBATCH -e sdpo_nofulllogit-%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH --mem=0

echo "Starting job on $(hostname)"
date

# ==========================================
# 1. CONFIGURATION & PORTS
# ==========================================
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
WRAPPER_SCRIPT="${WORK}/SDPO/tacc/generalization/ray_wrapper.sh"
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    WRAPPER_SCRIPT="${WORK}/SDPO/tacc/tests/ray_wrapper.sh"
fi
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found in tacc/generalization or tacc/tests"
    exit 1
fi
chmod +x "$WRAPPER_SCRIPT"

# ==========================================
# 2. HEAD NODE IDENTIFICATION
# ==========================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}

if [ -z "$HEAD_NODE" ]; then
    echo "Error: Could not determine HEAD_NODE from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    exit 1
fi

HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))")
fi

echo "Head Node: $HEAD_NODE"
echo "Head IP:   $HEAD_IP"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

echo "Cleaning up stale Ray sessions..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

# ==========================================
# 3. LAUNCH RAY CLUSTER
# ==========================================
echo "Launching Ray Head on $HEAD_NODE..."
numactl --cpunodebind=0 --membind=0 --physcpubind=0-71 "$WRAPPER_SCRIPT" ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=$RAY_PORT \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --min-worker-port=$RAY_MIN_WORKER_PORT \
    --max-worker-port=$RAY_MAX_WORKER_PORT \
    --num-gpus=1 \
    --block &
sleep 10

NUM_WORKERS=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= NUM_WORKERS; i++)); do
    node_i=${nodes_array[i]}
    echo "Starting Ray Worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "source ~/.bashrc && conda activate sdql && \
        numactl --cpunodebind=0 --membind=0 --physcpubind=0-71 '$WRAPPER_SCRIPT' ray start --address '$HEAD_IP:$RAY_PORT' --num-gpus=1 --num-cpus=32 --block" &
    sleep 5
done

echo "Waiting for Ray cluster to initialize..."
for i in {1..30}; do
    alive_nodes=$(ray status --address "$HEAD_IP:$RAY_PORT" 2>/dev/null | rg -c "node_" || true)
    if [ "${alive_nodes:-0}" -ge "$SLURM_JOB_NUM_NODES" ]; then
        echo "Ray Cluster Ready! ($alive_nodes nodes connected)"
        break
    fi
    sleep 5
done
ray status --address "$HEAD_IP:$RAY_PORT"

# ==========================================
# 4. TRAINING ARGUMENTS (from sbatch CLI)
# ==========================================
DATA_PATH=${1:-"datasets/sciknoweval/all"}
TRAIN_BATCH_SIZE=${2:-32}
ROLLOUT_BATCH_SIZE=${3:-8}
MINI_BATCH_SIZE=${4:-32}
LR=${5:-1e-5}
MODEL_PATH=${6:-"Qwen/Qwen3-8B"}
ALPHA=${7:-0.5}
DONT_REPROMPT_ON_SELF_SUCCESS=${8:-True}
SUFFIX=${9:-"local_sdpo_nofulllogit"}
TOPK=${10:-null}
TEACHER_UPDATE_RATE=${11:-0.05}
ENABLE_ICL=${12:-False}
CLAMP_HIGH=${13:-2.0}
CLAMP_LOW=${14:--2.0}
USE_REWARD_CLAMP=${15:-True}
IS_CLIP=${16:-2.0}
USE_FUTURE_RETURNS=${17:-False}
GAMMA=${18:-1.0}
USE_FUTURE_RETURNS_BASELINE=${19:-False}
CISPO_CLIP=${20:-0.2}

if [[ "${USE_REWARD_CLAMP}" == "True" || "${USE_REWARD_CLAMP}" == "true" || "${USE_REWARD_CLAMP}" == "1" ]]; then
    CLAMP_TAG="clamp${CLAMP_LOW}_${CLAMP_HIGH}"
else
    CLAMP_TAG="noclamp"
fi

if [[ "${ENABLE_ICL}" == "True" || "${ENABLE_ICL}" == "true" || "${ENABLE_ICL}" == "1" ]]; then
    CONFIG_NAME="sdpo_icl"
    # ICL training needs env feedback strings in the teacher prompt (use_feedback=True).
    INCLUDE_ENVIRONMENT_FEEDBACK=True
else
    CONFIG_NAME="sdpo"
    INCLUDE_ENVIRONMENT_FEEDBACK=False
fi



MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144}
TEMPLATE_LENGTH=512
MAX_FEEDBACK_LENGTH=$((MAX_RESPONSE_LENGTH + 512))
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_FEEDBACK_LENGTH + MAX_RESPONSE_LENGTH))
HOME_ROOT=${WORK}

export PROJECT_ROOT="${WORK}/SDPO"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export LOCAL_OUTPUT_DIR="${HOME_ROOT}/output/SDPO-nofulllogit-local"
CKPT_DIR="${LOCAL_OUTPUT_DIR}/checkpoints"
mkdir -p "$LOCAL_OUTPUT_DIR" "$CKPT_DIR"

export USER=${USER:-$(whoami)}
export TASK="${TASK:-$DATA_PATH}"
ICL_VALIDATION=${ICL_VALIDATION:-True}
ICL_VALIDATION_POLICY=${ICL_VALIDATION_POLICY:-teacher}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}
EXP_NAME="$SUFFIX"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
vars.dir=$PROJECT_ROOT \
trainer.experiment_name=$EXP_NAME \
vars.task=$TASK \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.log_dir=$LOCAL_OUTPUT_DIR \
vars.ckpt_dir=$CKPT_DIR \
trainer.group_name=SDPO-nofulllogit-generalization \
trainer.nnodes=8 \
trainer.n_gpus_per_node=1 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_MODEL_LEN \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
actor_rollout_ref.actor.self_distillation.distillation_topk=$TOPK \
actor_rollout_ref.actor.self_distillation.use_reward_clamp=$USE_REWARD_CLAMP \
actor_rollout_ref.actor.self_distillation.clamp_high=$CLAMP_HIGH \
actor_rollout_ref.actor.self_distillation.clamp_low=$CLAMP_LOW \
actor_rollout_ref.actor.self_distillation.is_clip=$IS_CLIP \
actor_rollout_ref.actor.self_distillation.cispo_clip=$CISPO_CLIP \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=${INCLUDE_ENVIRONMENT_FEEDBACK} \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
actor_rollout_ref.actor.self_distillation.use_future_returns=$USE_FUTURE_RETURNS \
actor_rollout_ref.actor.self_distillation.gamma=$GAMMA \
actor_rollout_ref.actor.self_distillation.use_future_returns_baseline=$USE_FUTURE_RETURNS_BASELINE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.icl_validation.enable=$ICL_VALIDATION \
trainer.icl_validation.policy=$ICL_VALIDATION_POLICY \
trainer.log_val_generations=$LOG_VAL_GENERATIONS \
trainer.test_freq=20 \
trainer.save_freq=20 \
trainer.max_actor_ckpt_to_keep=3 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "----------------------------------------------------------------"
echo "Starting SDPO no-full-logit Training"
echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG_NAME (enable_icl=$ENABLE_ICL, include_environment_feedback=$INCLUDE_ENVIRONMENT_FEEDBACK)"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Alpha: $ALPHA"
echo "full_logit_distillation: False"
echo "is_clip: $IS_CLIP"
echo "cispo_clip: $CISPO_CLIP"
echo "Clamp: [$CLAMP_LOW, $CLAMP_HIGH] (use_reward_clamp=$USE_REWARD_CLAMP, tag=$CLAMP_TAG)"
echo "use_future_returns: $USE_FUTURE_RETURNS (gamma=$GAMMA)"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
