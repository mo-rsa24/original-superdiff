## What’s different (conceptually)

### 1) is mostly *surgical + mechanics of composition*

It reads like: “keep the expert FCN, widen receptive field minimally, make Regime-C overlap actually happen, and fix/normalize PoE sampling dynamics.”

* **Model change:** *Multi-dilated bottleneck + residual follow-ups* to expand receptive field **without breaking FCN/translation equivariance**.
* **Training/data regime change:** Explicitly **bias sampling to force overlap** between experts (push extra digits toward the *opposing* expert’s digit). This is about **joint support coverage**, not just capacity.
* **Sampler change:** **Energy/scale balancing** in PoE at DDIM time (optional renormalization of combined noise estimate). This is about **composition stability/calibration**.

So (1) is: *“PoE composition reliability via overlap engineering + sampler scale calibration, with minimal architecture change.”*

---

### 2) is more *representation + systemization*

It reads like: “upgrade the expert so it can represent ‘existential semantics’ better, then pick stable Regime-C defaults and add deep auditing/diagnostics.”

* **Model change:** Bigger, more expressive FCN: *dilation-aware residual blocks + translation-equivariant spatial attention + receptive field reporting*. This is **capacity/semantics** oriented.
* **Regime defaults:** Defines a **reference recipe** (deeper dilated stack, attention mid-run, dropout, more steps) tuned for “expected-success” PoE generation. This is about **a stable canonical configuration**, not a knob-by-knob patch.
* **Pipeline/auditing:** Diagnostics, logging fixes, hyperparameter threading for reliable evaluation. This is about **reproducibility + observability**.

So (2) is: *“Semantic capacity + stable canonical regimen + instrumentation.”*

---

## How I’d name them as branches under a shared base branch

### Base branch (shared umbrella)

**FCN PoE Regime-C (Compositional Diffusion)**
or more descriptive: **Translation-Equivariant FCN Experts for Regime-C PoE Composition**

---

## Branch taxonomy suggestion

### Branch A (for idea set 1)

**Overlap-Calibrated PoE Composition**

* alt names:

  * **Regime-C Overlap & Sampler Calibration**
  * **PoE Joint-Support Engineering**
  * **Composition Mechanics: Overlap + Energy Normalization**
* why this fits: it’s primarily about **making composition work** (data overlap + sampling scale), with a minimal receptive-field tweak.

### Branch B (for idea set 2)

**Semantics-Enhanced Expert Architecture & Audited Regime-C**

* alt names:

  * **Attention-Augmented FCN Experts**
  * **Representation + Defaults + Diagnostics**
  * **Regime-C Reference Stack (Dilations+TE-Attention)**
* why this fits: it’s primarily about **improving what the expert can represent** + defining a **stable default configuration** + **instrumentation**.

---

## If you want crisp “branch labels” that read well in a repo

* `base/poe-regime-c-fcn`
* `branch/poe-overlap-calibration`  *(idea set 1)*
* `branch/fcn-attention-semantics-audit` *(idea set 2)*

Or shorter:

* `regime-c-overlap-cal`
* `regime-c-semantics-audit`

---

If I had to summarize the distinction in one line:

* **(1) makes PoE composition behave** (overlap + scale calibration).
* **(2) makes the expert understand better and makes the pipeline trustworthy** (semantics capacity + stable defaults + diagnostics).
