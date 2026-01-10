# Experiment Regime B: The Inductive Bias Mismatch

## 1. Abstract
Regime B is the **critical negative control for architecture**. It investigates a subtle failure mode where the *dataset* allows composition (overlapping support), but the *model architecture* implicitly forbids it. This regime proves that "data availability" is insufficient if the model's inductive bias enforces a "single-object" or "center-focused" constraint.

## 2. Theoretical Formulation

### 2.1 The Data Manifold
The dataset contains digits placed randomly on a canvas $x \in \mathbb{R}^{H \times W}$. The true distribution $p_{data}(x)$ is **translation-invariant** and **existential**:
$$p(x) = \int p(x|z, \text{loc}) p(\text{loc}) d\text{loc}$$
Crucially, $p_{joint}(x)$ (two digits) is valid and exists within the pixel space.

### 2.2 The Architectural Conflict (Center Bias)
The experts in Regime B utilize a U-Net variant with a **Spatial Bottleneck**:
* **Operation:** $h_{bottleneck} = \text{GlobalAveragePool}(E(x))$ or $h_{bottleneck} \in \mathbb{R}^{d}$ (1D vector).
* **Consequence:** The model loses spatial resolution at the deepest layer, forcing it to compress the input into a single semantic vector $z$.

This imposes a **Unimodal Spatial Assumption**:
$$q_\theta(x) \approx \mathcal{N}(x; \mu_{center}, \sigma)$$
The model learns the "average" digit centered in the image because it cannot resolve the spatial uncertainty of the translation-invariant dataset.

### 2.3 The "Mean Collapse" Failure
When trained on location-agnostic data, a spatially constrained model converges to the **Barycenter of the Dataset**:
$$\min_\theta \mathbb{E}_{x \sim p_{data}} [ || \epsilon_\theta(x_t) - \epsilon ||^2 ] \implies \epsilon_\theta(x_t) \approx \mathbb{E}[ \text{noise across all locations} ]$$
The resulting samples are "blurry means" or high-entropy texture because the model is averaging mutually exclusive spatial possibilities (a '4' at top-left vs. a '4' at bottom-right).

## 3. Experimental Validation
* **Observed Outcome:** High-frequency noise, lack of structure, failure to converge to any digit.
* **Why Valid:** This confirms the architecture cannot model the *marginal* $p(x|c)$, let alone the composition.
* **Conclusion:** Verified. Overlapping data support is insufficient without **Translation-Equivariant** inductive biases.