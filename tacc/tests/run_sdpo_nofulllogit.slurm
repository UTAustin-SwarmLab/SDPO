#!/bin/bash
#SBATCH -J sdpo_nofulllogit
#SBATCH -o sdpo_nofulllogit-%j.out
#SBATCH -e sdpo_nofulllogit-%j.err
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH --mem=0

# Usage: sbatch tacc/tests/run_sdpo_nofulllogit.slurm
# Purpose: SDPO smoke test with sampled-token (non-full-logit) distillation
#
# Optional env overrides:
#   ALPHA=0.0|0.5|1.0  DATA_PATH=...  MODEL_PATH=...  EXP_NAME=...

echo "Starting SDPO no-full-logit smoke test on $(hostname)"
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

CONFIG_NAME="sdpo"
DATA_PATH="${DATA_PATH:-datasets/sciknoweval/all}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}"
CLAMP_HIGH="${CLAMP_HIGH:-2.0}"
CLAMP_LOW="${CLAMP_LOW:--2.0}"
USE_REWARD_CLAMP="${USE_REWARD_CLAMP:-True}"
IS_CLIP="${IS_CLIP:-0.2}"
USE_GAE="${USE_GAE:-False}"
GAMMA="${GAMMA:-1.0}"
DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-True}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TEMPLATE_LENGTH=512
MAX_FEEDBACK_LENGTH=$((MAX_RESPONSE_LENGTH + 512))
MAX_MODEL_LEN=$((TEMPLATE_LENGTH + MAX_PROMPT_LENGTH + MAX_FEEDBACK_LENGTH + MAX_RESPONSE_LENGTH))

if [[ "${USE_REWARD_CLAMP}" == "True" || "${USE_REWARD_CLAMP}" == "true" || "${USE_REWARD_CLAMP}" == "1" ]]; then
    CLAMP_TAG="clamp${CLAMP_LOW}_${CLAMP_HIGH}"
else
    CLAMP_TAG="noclamp"
fi

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export TASK="${TASK:-$DATA_PATH}"
ICL_VALIDATION=${ICL_VALIDATION:-True}
ICL_VALIDATION_POLICY=${ICL_VALIDATION_POLICY:-teacher}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}
OUTPUT_DIR="${WORK}/output/SDPO-nofulllogit-test"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

EXP_NAME="${EXP_NAME:-SMOKE-SDPO-nofulllogit-${CLAMP_TAG}-a${ALPHA}-$(date +%Y%m%d-%H%M%S)}"
ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$TRAIN_BATCH_SIZE \
trainer.experiment_name=$EXP_NAME \
vars.task=$TASK \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
vars.dir=$PROJECT_ROOT \
vars.log_dir=$OUTPUT_DIR \
vars.ckpt_dir=$CKPT_DIR \
trainer.group_name=SDPO-nofulllogit-test \
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
actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
actor_rollout_ref.actor.self_distillation.distillation_topk=null \
actor_rollout_ref.actor.self_distillation.use_reward_clamp=$USE_REWARD_CLAMP \
actor_rollout_ref.actor.self_distillation.clamp_high=$CLAMP_HIGH \
actor_rollout_ref.actor.self_distillation.clamp_low=$CLAMP_LOW \
actor_rollout_ref.actor.self_distillation.is_clip=$IS_CLIP \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.use_gae=$USE_GAE \
actor_rollout_ref.actor.self_distillation.gamma=$GAMMA \
actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.val_kwargs.n=2 \
trainer.icl_validation.enable=$ICL_VALIDATION \
trainer.icl_validation.policy=$ICL_VALIDATION_POLICY \
trainer.log_val_generations=$LOG_VAL_GENERATIONS \
trainer.test_freq=20 \
trainer.save_freq=20 \
trainer.max_actor_ckpt_to_keep=3 \
actor_rollout_ref.rollout.agent.num_workers=2 \
actor_rollout_ref.rollout.max_num_seqs=32 \
trainer.val_before_train=False \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "Starting SDPO no-full-logit smoke-test training run..."
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Alpha: $ALPHA"
echo "Clamp: [$CLAMP_LOW, $CLAMP_HIGH] (use_reward_clamp=$USE_REWARD_CLAMP, tag=$CLAMP_TAG)"
echo "is_clip: $IS_CLIP"
echo "full_logit_distillation: False"
echo "use_gae: $USE_GAE (gamma=$GAMMA)"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
