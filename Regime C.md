### Branching strategy for “two separate ideas”
Create one long-lived base branch per regime (e.g., `regime-c-base`). This holds the shared fixes you want in both ideas.

For each idea, branch off `regime-c-base`:
- `regime-c-idea1-attn-dilate`
- `regime-c-idea2-largekernels`

Keep branches short-lived; merge only when an idea proves better, or tag the tip commit instead of merging if you want a permanent marker without mixing code paths.

Git example:
```bash 
git checkout main
git checkout -b regime-c-base
# apply common fixes, commit
git checkout -b regime-c-idea1-attn-dilate
# apply idea 1 changes, commit
git checkout regime-c-base
git checkout -b regime-c-idea2-largekernels
# apply idea 2 changes, commit
```

General (works for any branch)
```bash
export WANDB_NAME="regimeC-idea1-$(git rev-parse --short HEAD)"
export WANDB_TAGS="regimeC,idea1,$(git rev-parse --short HEAD)"
./cifar/launch_train.sh C --workdir "cifar/runs/regime_C_idea1"
```

To Run Idea 1: `regime-c-idea-1-poe-overlap-calibration`

```bash
git checkout regime-c-idea-1-poe-overlap-calibration
export WANDB_NAME="idea1-$(git rev-parse --short HEAD)"
export WANDB_TAGS="regimeC,idea1,$(git rev-parse --short HEAD)"
./cifar/launch_train.sh C --workdir "cifar/runs/regime_C_idea1"
```

To Run Idea 2: `regime-c-idea-2-fcn-attention-semantics`
```bash
git checkout regime-c-idea-2-fcn-attention-semantics
export WANDB_NAME="idea2-$(git rev-parse --short HEAD)"
export WANDB_TAGS="regimeC,idea2,$(git rev-parse --short HEAD)"
./cifar/launch_train.sh C --workdir "cifar/runs/regime_C_idea2"

```

**W&B:** Filter by tags (regimeC, idea1, commit hash) or run name you set; runs are grouped under WANDB_RUN_GROUP (default regime_C).

**Post-run mapping:** Refer to logs/experiments.md for SLURM→W&B linkage, or add more context manually if desired.