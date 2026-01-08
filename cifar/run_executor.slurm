#!/usr/bin/env bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=mscluster65,mscluster76
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# --- 1. Environment Setup ---
# Ensure log directory exists
mkdir -p logs
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/cifar" ]]; then
  cd "${SLURM_SUBMIT_DIR}/cifar"
else
  cd "$SCRIPT_DIR"
fi
# Load Conda/Mamba (Adjust path to your shell config if needed)
set +u
source ~/.bashrc

echo "Activate environment: ${ENV_NAME:-jaxstack}"
mamba activate "${ENV_NAME:-jaxstack}"
set -u

# --- 2. Execution Info ---
echo "========================================"
echo "Starting Job on Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Arguments: $@"
echo "Commit: ${GIT_COMMIT_SHORT:-unknown} | Branch: ${GIT_BRANCH:-unknown}"
echo "========================================"

# --- 3. Run Code ---
# Passes all arguments received from launcher directly to main.py
# -u forces unbuffered output so logs appear instantly
# Prevent JAX from grabbing 90% of GPU memory at startup (crucial for sharing GPUs or debugging)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# If using TensorFlow dataloaders, prevent them from competing for GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
echo "PYTHONPATH set to: $PYTHONPATH"
srun python -u main.py "$@"