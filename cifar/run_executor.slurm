#!/usr/bin/env bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=mscluster65,mscluster76
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# --- 1. Environment Setup ---
mkdir -p logs

# Prefer running from the staged repo (SLURM_SUBMIT_DIR is the staging dir used by launch_train.sh)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/cifar" ]]; then
  cd "${SLURM_SUBMIT_DIR}/cifar"
elif [[ -d "cifar" ]]; then
  cd "cifar"
else
  echo "ERROR: Could not locate cifar/ directory. PWD=$(pwd)"
  exit 1
fi

# --- 2. Execution Info ---
echo "========================================"
echo "Starting Job on Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Arguments: $*"
echo "Commit: ${GIT_COMMIT_SHORT:-unknown} | Branch: ${GIT_BRANCH:-unknown}"
if [[ -n "${WANDB_PROJECT:-}" ]]; then echo "W&B Project: ${WANDB_PROJECT}"; fi
if [[ -n "${WANDB_NAME:-}" ]]; then echo "W&B Name: ${WANDB_NAME}"; fi
if [[ -n "${WANDB_TAGS:-}" ]]; then echo "W&B Tags: ${WANDB_TAGS}"; fi
if [[ -n "${WANDB_RUN_GROUP:-}" ]]; then echo "W&B Group: ${WANDB_RUN_GROUP}"; fi
echo "========================================"

# --- 3. Runtime env ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
echo "PYTHONPATH set to: $PYTHONPATH"

# Make git metadata available to python even if you don't pass explicit flags
export GIT_BRANCH="${GIT_BRANCH:-unknown}"
export GIT_COMMIT_SHORT="${GIT_COMMIT_SHORT:-unknown}"

# --- 4. Run Code ---
# Passes all args received from launcher directly to main.py
srun python -u main.py "$@"
