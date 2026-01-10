#!/usr/bin/env bash
set -euo pipefail

# Using tput for better compatibility and cleaner syntax
if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

# --- 1. Define Defaults ---
export ENV_NAME="${ENV_NAME:-jaxstack}"
export SLURM_PARTITION="${SLURM_PARTITION:-bigbatch}"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"

# Usage: ./launch_train.sh [A|B|C] [--partition p] [--time hh:mm:ss] [--workdir path] [--wandb_project name] [--wandb_name name]
REGIME=${1:-A}
shift || true
REGIME_LOWER=$(echo "$REGIME" | tr '[:upper:]' '[:lower:]')

REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

export CONFIG="${CONFIG:-cifar/configs/sm/mnist/regime_${REGIME_LOWER}.py}"
export WORKDIR="${WORKDIR:-cifar/runs/regime_${REGIME}}"
export WANDB_PROJECT="${WANDB_PROJECT:-ebm-rq1-run-3}"
export WANDB_NAME="${WANDB_NAME:-${REGIME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export WANDB_TAGS="${WANDB_TAGS:-regime_${REGIME},${GIT_BRANCH},${GIT_HASH}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-regime_${REGIME}}"

JOB_NAME="rq1-regime-${REGIME}-${GIT_HASH}"
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${TIMESTAMP}"

# --- 3. Parse Command Line Overrides (Optional) ---
# Allows overriding the defaults if needed
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition) export SLURM_PARTITION="$2"; shift 2 ;;
    --time)      export TIME_LIMIT="$2"; shift 2 ;;
    --workdir)   export WORKDIR="$2"; shift 2 ;;
    --wandb_project) export WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_name) export WANDB_NAME="$2"; shift 2 ;;
    --wandb_tags) export WANDB_TAGS="$2"; shift 2 ;;
    --wandb_group) export WANDB_RUN_GROUP="$2"; shift 2 ;;
    *) shift ;;
  esac
done

header "🚀 Snapshotting & Submitting"

# --- 4. Perform the Snapshot ---
status_line "📦 Staging To" "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

rsync -a \
  --exclude 'logs' \
  --exclude 'cifar/runs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'wandb' \
  "$REPO_ROOT/" "$STAGING_DIR/"

status_line "✅ Snapshot" "Complete"


status_line "--------------------------------------------------------"
# --- 5. Submit from the Staging Directory ---
cd "$STAGING_DIR"

status_line "--------------------------------------------------------"
status_line "📜 Config"  "$CONFIG"
status_line "📂 Workdir" "$WORKDIR"
status_line "📌 Commit"  "$GIT_HASH"
status_line "🏷️  Branch"  "$GIT_BRANCH"
status_line "🪪 W&B"     "$WANDB_NAME"

mkdir -p "${REPO_ROOT}/logs"
JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
       --job-name="$JOB_NAME" \
       --time="$TIME_LIMIT" \
       --output="${REPO_ROOT}/logs/%x-%j.out" \
       --error="${REPO_ROOT}/logs/%x-%j.err" \
       --export=ALL,ENV_NAME="$ENV_NAME",GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",WANDB_NAME="$WANDB_NAME",WANDB_TAGS="$WANDB_TAGS",WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_PROJECT="$WANDB_PROJECT" \
       cifar/run_executor.slurm \
       --config "$CONFIG" \
       --workdir "$WORKDIR" \
       --mode train \
       --sample_every 10000 \
       --wandb_project "$WANDB_PROJECT" | awk '{print $4}')

status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at" "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"

# Record experiment provenance locallys
mkdir -p "${REPO_ROOT}/logs"
{
  echo "- ${TIMESTAMP} | job: ${JOB_NAME} | id: ${JOB_ID} | branch: ${GIT_BRANCH} | commit: ${GIT_HASH} | wandb_name: ${WANDB_NAME} | wandb_project: ${WANDB_PROJECT} | workdir: ${WORKDIR}"
} >> "${REPO_ROOT}/logs/experiments.md"

# Return to original dir
cd "$REPO_ROOT"