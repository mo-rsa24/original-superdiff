#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=mscluster65,mscluster76

set -euo pipefail


# --- 1. Environment Setup ---
mkdir -p logs
set +u
source ~/.bashrc
# Activate environment (Defaults to jax115 if not set)
echo "Activating Environment: ${ENV_NAME:-jaxstack}"
mamba activate "${ENV_NAME:-jaxstack}"
set -u

# --- 2. Execution Flags ---
# Prevent JAX and TF from pre-allocating all VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- 3. Run rq1.py ---
# All arguments passed from sbatch (in launch_train.sh) are caught by "$@"
echo "Executing: python -u rq1.py $@"
echo "Commit: ${GIT_COMMIT_SHORT:-unknown} | Branch: ${GIT_BRANCH:-unknown} | W&B Name: ${WANDB_NAME:-unset}"
srun python3 -u cifar/rq1.py "$@"