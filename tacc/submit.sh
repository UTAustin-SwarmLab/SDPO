#!/bin/bash

TACC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tacc/config.sh
source "${TACC_DIR}/config.sh"

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"
    local output_subdir="$4"

    local output_dir="${LOG_DIR}/${output_subdir}"
    mkdir -p "$output_dir"

    local job_script="${JOBS_DIR}/${exp_name}.slurm"
    cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH -J ${BASE_JOB_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NODES}
#SBATCH -n ${NTASKS}
#SBATCH -t ${TIME}
#SBATCH -o ${output_dir}/${exp_name}-%j.out
#SBATCH -e ${output_dir}/${exp_name}-%j.err

set -euo pipefail

export EXP_NAME="${exp_name}"
export CONFIG_NAME="${CONFIG_NAME}"
export DATA_PATH="${data_path}"
export SCRIPT_ARGS="${script_args}"
export OUTPUT_SUBDIR="${output_subdir}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"
export TACC_PROJECT_ROOT="${PROJECT_ROOT}"
export TACC_LOG_DIR="${LOG_DIR}"
export TACC_CKPT_DIR="${CKPT_DIR}"

bash "${TACC_DIR}/run_training.sh"
EOF

    if [[ "${DRY_RUN:-false}" == true ]]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: ${exp_name}"
        echo "Job script: ${job_script}"
        cat "$job_script"
    else
        echo "Submitting job for: ${exp_name}"
        sbatch "$job_script"
    fi
}
