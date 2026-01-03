#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=mscluster65,mscluster76

set -euo pipefail

# --- 1. Environment Setup ---
mkdir -p logs
source ~/.bashrc

# Activate environment (Defaults to jax115 if not set)
echo "Activating Environment: ${ENV_NAME:-jaxstack}"
set +u
mamba activate "${ENV_NAME:-jaxstack}"
set -u

# --- 2. Execution Flags ---
# Prevent JAX and TF from pre-allocating all VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

# --- 3. Run rq1.py ---
# All arguments passed from sbatch (in launch_train.sh) are caught by "$@"
echo "Executing: python -u rq1.py $@"
srun python3 -u cifar.rq1.py "$@"