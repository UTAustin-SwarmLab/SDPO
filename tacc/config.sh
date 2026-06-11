#!/bin/bash
# TACC cluster settings for SDPO experiment sweeps.
# Edit ACCOUNT and paths before submitting jobs.

# Slurm allocation (required on TACC when you belong to multiple projects)
ACCOUNT="${TACC_ACCOUNT:ASC26054}"

# Vista GPU queue; use gpu-a100, gpu-h100, etc. on other TACC systems
PARTITION="${TACC_PARTITION:-gh}"
TIME="${TACC_TIME:-12:00:00}"
BASE_JOB_NAME="${TACC_JOB_NAME:-rlvr}"

# Vista has 1 GH200 GPU per node (no --gres / --gpus-per-node).
# For CSCS-equivalent 4-GPU runs, set NODES=4 and NTASKS=4 (Ray cluster required).
NODES="${TACC_NODES:-1}"
NTASKS="${TACC_NTASKS:-1}"
GPUS_PER_NODE="${TACC_GPUS_PER_NODE:-1}"

# Repository and artifact paths
PROJECT_ROOT="${TACC_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${TACC_LOG_DIR:-${WORK}/SDPO/output}"
CKPT_DIR="${TACC_CKPT_DIR:-${WORK}/SDPO/checkpoints}"
JOBS_DIR="${TACC_JOBS_DIR:-${PROJECT_ROOT}/tacc/jobs}"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$JOBS_DIR"
