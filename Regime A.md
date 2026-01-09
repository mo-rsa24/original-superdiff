# Experiment Regime A: The Support Mismatch Hypothesis

## 1. Abstract
Regime A serves as the **negative control for data support**. It tests the fundamental prerequisite of Product-of-Experts (PoE) composition: that the component distributions must have a non-null intersection in high-dimensional pixel space. We demonstrate that even with perfect architectural inductive biases, composition is mathematically impossible if the expert manifolds are disjoint.

## 2. Theoretical Formulation

### 2.1 The Compositional Goal
We aim to sample from the joint distribution $p_{joint}(x)$ satisfying conditions $c_1$ and $c_2$:
$$p_{joint}(x) \propto p(x|c_1) \cdot p(x|c_2)$$
or in terms of Energy-Based Models (EBMs):
$$E_{joint}(x) = E(x|c_1) + E(x|c_2)$$

### 2.2 The Disjoint Support Condition
In Regime A, the experts are trained on datasets $\mathcal{D}_1$ and $\mathcal{D}_2$ such that their approximate supports $\mathcal{S}_1$ and $\mathcal{S}_2$ satisfy:
$$\mathcal{S}_1 \cap \mathcal{S}_2 \approx \emptyset$$

**Implementation:**
* **Expert 1 ($c_1$):** Trained on "Left-Half" MNIST digits (right half masked).
* **Expert 2 ($c_2$):** Trained on "Right-Half" MNIST digits (left half masked).

### 2.3 The Failure Mode
Since the score function $\nabla_x \log p(x)$ is only well-defined within the support of the data, the compositional gradient field becomes ill-posed:
$$\nabla_x E_{joint}(x) = \underbrace{\nabla_x E_1(x)}_{\approx 0 \text{ or undefined}} + \underbrace{\nabla_x E_2(x)}_{\approx 0 \text{ or undefined}}$$
In regions where $p_1(x) \approx 0$, the score is erratic or pushes $x$ towards $\mathcal{S}_1$, which is disjoint from $\mathcal{S}_2$. The Langevin dynamics settle into low-likelihood "bridges" or noise.

## 3. Experimental Validation
* **Observed Outcome:** Samples fail to merge concepts; outputs resemble linear superpositions (ghosting) or pure noise.
* **Conclusion:** Verified. Compositional diffusion requires intersecting data supports.