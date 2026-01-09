# Experiment Regime C: Compositional Success via Spatial Equivariance

## 1. Abstract
Regime C represents the **positive control** or the "working system." It combines the overlapping data support of Regime B with an architecture designed for **Existential Semantics**. It demonstrates that when inductive biases align with the logical structure of the task, energy-based composition succeeds zero-shot.

## 2. Theoretical Formulation

### 2.1 The Existential Architecture
The experts use a **Fully Convolutional U-Net** (or Attention with appropriate positional embeddings) that preserves spatial dimensions throughout the bottleneck:
$$h_{bottleneck} \in \mathbb{R}^{h \times w \times d}$$
There is no global pooling. This ensures **Translation Equivariance**:
$$f(T_g(x)) = T_g(f(x))$$
where $T_g$ is a spatial shift.

### 2.2 Localized Energy Potentials
Because the architecture preserves locality, the learned energy function $E(x)$ decomposes spatially:
$$E(x) \approx \sum_{i,j} E_{local}(x_{i,j})$$
This allows the model to represent "a digit '4' exists at $(u,v)$" independently of what exists at $(u', v')$.

### 2.3 Composition as Logic
When composing gradients:
$$\nabla_x E_{joint} = \nabla_x E_{\text{concept 1}} + \nabla_x E_{\text{concept 2}}$$
The spatially localized activations allows the two gradients to operate on different regions of the canvas without destructive interference.
* Expert 1 guides pixels at $(u,v)$ to form a '4'.
* Expert 2 guides pixels at $(u',v')$ to form a '7'.
* **Result:** A coherent image satisfying $c_1 \wedge c_2$.

## 3. Experimental Validation
* **Observed Outcome:** Clean samples containing both digits at distinct locations.
* **Conclusion:** Validated. The combination of **Overlapping Support** + **Spatial Equivariance** is the sufficient condition for compositional generation.