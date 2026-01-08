#!/usr/bin/env bash
set -euo pipefail

# --- 1. Define Defaults (Environment & SLURM) ---
export ENV_NAME="${ENV_NAME:-jax115}"
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


# --- 3. Pretty Print Helpers (Replicated from train_ldm_conditional.sh) ---
if [[ -t 1 ]]; then
  BLD=$(tput bold); CYN=$(tput setaf 6); BLU=$(tput setaf 4); RST=$(tput sgr0)
else
  BLD=""; CYN=""; BLU=""; RST=""
fi

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
rule
printf "${BLD}${BLU}🚀 Submitting SuperDiff Job: ${JOB_NAME}${RST}\n"
rule
kv "SLURM Partition" "${SLURM_PARTITION}"
kv "Time Limit"      "${TIME_LIMIT}"
kv "Log Directory"   "logs/${JOB_NAME}"
printf "\n"
kv "🐍 Environment"  "${ENV_NAME}"
kv "📜 Config"       "${CONFIG}"
kv "📂 Workdir"      "${WORKDIR}"
kv "⚙️ Mode"         "${MODE}"
kv "🌿 Branch"       "${GIT_BRANCH}"
kv "🔖 Commit"       "${GIT_HASH}"
if [[ -n "$WANDB_ID" ]]; then
    kv "📈 W&B ID"   "${WANDB_ID}"
fi
rule

# --- 6. Snapshot to Staging (avoid git race conditions on cluster) ---
kv "📦 Staging To" "${STAGING_DIR}"
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

kv "🎉 Submitted" "Job ID: ${JOB_ID}"
kv "📝 Logs at"   "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"
cd "$REPO_ROOT"