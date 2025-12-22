#!/usr/bin/env bash
set -euo pipefail

# --- 1. Define Defaults (Environment & SLURM) ---
export ENV_NAME="jax115"
export SLURM_PARTITION="bigbatch"
export TIME_LIMIT="72:00:00"
export JOB_NAME="superdiff-cifar"

# --- 2. Project Defaults (Specific to SuperDiff/VPSDE) ---
# Default config points to the file identified in your codebase
export CONFIG="cifar/configs/sm/cifar/vpsde.py"
export WORKDIR="exp_output"
export MODE="train"
export WANDB_ID=""

# --- 3. Pretty Print Helpers (Replicated from train_ldm_conditional.sh) ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
rule(){ printf "${BLU}%.0s" $(seq 1 60); printf "${RST}\n"; }

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
    *)           echo "Unknown argument: $1"; shift ;;
  esac
done

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
if [[ -n "$WANDB_ID" ]]; then
    kv "📈 W&B ID"   "${WANDB_ID}"
fi
rule

# --- 6. Submit to SLURM ---
# We pass the python arguments (flags) to the executor script
sbatch --partition="$SLURM_PARTITION" \
       --job-name="$JOB_NAME" \
       --time="$TIME_LIMIT" \
       --output="logs/%x-%j.out" \
       --error="logs/%x-%j.err" \
       run_executor.slurm \
       --config "$CONFIG" \
       --workdir "$WORKDIR" \
       --mode "$MODE" \
       ${WANDB_ID:+--wandb_id "$WANDB_ID"}