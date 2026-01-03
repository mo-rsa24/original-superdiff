#!/usr/bin/env bash
set -euo pipefail

# Using tput for better compatibility and cleaner syntax
BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RESET=$(tput sgr0)

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "$2"; }
header()      { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; printf "${BLUE}%.0s-$(seq 1 50)${RESET}\n"; }

# --- 1. Define Defaults (Environment & SLURM) ---
export ENV_NAME="jaxstack"
export SLURM_PARTITION="bigbatch"
export TIME_LIMIT="72:00:00"

# --- 2. Parse Regime and Setup Paths ---
# Usage: ./launch_train.sh [A|B|C]
REGIME=${1:-A}
shift
REGIME_LOWER=$(echo "$REGIME" | tr '[:upper:]' '[:lower:]')

export JOB_NAME="rq1-regime-${REGIME}"
export CONFIG="cifar/configs/sm/mnist/regime_${REGIME_LOWER}.py"
export WORKDIR="cifar/runs/regime_${REGIME}"
export WANDB_PROJECT="ebm-rq1"

# --- 3. Parse Command Line Overrides (Optional) ---
# Allows overriding the defaults if needed
while [[ $# -gt 1 ]]; do
  case $1 in
    --partition) export SLURM_PARTITION="$2"; shift 2 ;;
    --time)      export TIME_LIMIT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# --- 4. Print Submission Banner ---

status_line "========================================================"
header "🚀 Submitting RQ1 Job: ${JOB_NAME}"
status_line "--------------------------------------------------------"
status_line "📜 Config:  $CONFIG"
status_line "📂 Workdir: $WORKDIR"
status_line "⚙️ Mode:    train"
status_line "📈 Project: $WANDB_PROJECT"
status_line "========================================================"
printf "${BLUE}%.0s-$(seq 1 50)${RESET}\n"
# --- 5. Submit to SLURM ---
# Pass the regime-specific variables and flags to run_executor.slurm
sbatch --partition="$SLURM_PARTITION" \
       --job-name="$JOB_NAME" \
       --time="$TIME_LIMIT" \
       --output="logs/%x-%j.out" \
       --error="logs/%x-%j.err" \
       --export=ALL,ENV_NAME=$ENV_NAME \
       cifar/run_executor.slurm \
       --config "$CONFIG" \
       --workdir "$WORKDIR" \
       --mode train \
       --wandb_project "$WANDB_PROJECT" \
       --sample_every 5000