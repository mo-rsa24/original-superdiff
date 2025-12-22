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
srun python -u main.py "$@"