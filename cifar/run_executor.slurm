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

# Load Conda/Mamba (Adjust path to your shell config if needed)
source ~/.bashrc

echo "Activate environment: ${ENV_NAME:-jax115}"
mamba activate "${ENV_NAME:-jax115}"

# --- 2. Execution Info ---
echo "========================================"
echo "Starting Job on Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Arguments: $@"
echo "========================================"

# --- 3. Run Code ---
# Passes all arguments received from launcher directly to main.py
# -u forces unbuffered output so logs appear instantly
# Prevent JAX from grabbing 90% of GPU memory at startup (crucial for sharing GPUs or debugging)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# If using TensorFlow dataloaders, prevent them from competing for GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
srun python -u main.py "$@"