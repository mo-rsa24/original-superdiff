#!/usr/bin/env bash
set -euo pipefail

# --- 1. Define Defaults (Environment & SLURM) ---
export ENV_NAME="${ENV_NAME:-jaxstack}"
export SLURM_PARTITION="${SLURM_PARTITION:-bigbatch}"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"
export JOB_NAME="${JOB_NAME:-superdiff-cifar}"

# --- 2. Project Defaults (Specific to SuperDiff/VPSDE) ---
# Default config points to the file identified in your codebase
export CONFIG="${CONFIG:-configs/sm/cifar/vpsde.py}"
export WORKDIR="${WORKDIR:-exp_output}"
export MODE="${MODE:-train}"
export WANDB_ID="${WANDB_ID:-}"
export USE_WANDB="${USE_WANDB:-}"


if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

# --- 4. Parse Command Line Overrides ---
# Allows: ./launch_train.sh --partition stampede --config my_config.py
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition) export SLURM_PARTITION="$2"; shift 2 ;;
    --name)      export JOB_NAME="$2"; shift 2 ;;
    --time)      export TIME_LIMIT="$2"; shift 2 ;;
    # Project specific args
    --config)    export CONFIG="$2"; shift 2 ;;
    --workdir)   export WORKDIR="$2"; shift 2 ;;
    --mode)      export MODE="$2"; shift 2 ;;
    --wandb-id)  export WANDB_ID="$2"; shift 2 ;;
    --use_wandb) export USE_WANDB="--use_wandb"; shift ;;
    *)           echo "Unknown argument: $1"; shift ;;
  esac
done

REPO_ROOT=$(git rev-parse --show-toplevel)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${GIT_HASH}_${TIMESTAMP}"

# --- 5. Print Submission Banner ---
header "🚀 Submitting SuperDiff Job: ${JOB_NAME}"
status_line "SLURM Partition" "${SLURM_PARTITION}"
status_line "Time Limit"      "${TIME_LIMIT}"
status_line "Log Directory"   "logs/${JOB_NAME}"
rule
status_line "🐍 Environment"  "${ENV_NAME}"
status_line "📜 Config"       "${CONFIG}"
status_line "📂 Workdir"      "${WORKDIR}"
status_line "⚙️ Mode"         "${MODE}"
status_line "🌿 Branch"       "${GIT_BRANCH}"
status_line "🔖 Commit"       "${GIT_HASH}"
if [[ -n "$WANDB_ID" ]]; then
  status_line "📈 W&B ID" "${WANDB_ID}"
fi
rule


# --- 6. Snapshot to Staging (avoid git race conditions on cluster) ---
status_line "📦 Staging To" "${STAGING_DIR}"
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' \
  --exclude 'cifar/runs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'wandb' \
  "$REPO_ROOT/" "$STAGING_DIR/"

# --- 7. Submit to SLURM (from staging dir) ---
mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/cifar/runs"
JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
       --job-name="$JOB_NAME" \
       --time="$TIME_LIMIT" \
       --output="${REPO_ROOT}/logs/%x-%j.out" \
       --error="${REPO_ROOT}/logs/%x-%j.err" \
       --export=ALL,ENV_NAME="$ENV_NAME",GIT_COMMIT_SHORT="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH" \
       cifar/run_executor.slurm \
       --config "$CONFIG" \
       --workdir "$WORKDIR" \
       --mode "$MODE" \
       ${WANDB_ID:+--wandb_id "$WANDB_ID"} \
       $USE_WANDB | awk '{print $4}')

status_line "🎉 Submitted" "Job ID: ${JOB_ID}"
status_line "📝 Logs at"   "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"
cd "$REPO_ROOT"