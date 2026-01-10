# Experiment 1 — Logical AND of Two Shapes (4 ∧ 7)

## Research Question

Can diffusion-based experts composed via a product of experts generate **spatially separated co-occurring objects** without explicit spatial constraints?

## Goals and Objectives

**Goal.**
To empirically test when and why **product-of-experts (PoE) composition** of diffusion models—interpreted as energy-based models—can successfully generate samples satisfying the **logical conjunction** of two concepts (digit *4* ∧ digit *7*).

**Primary Objective.**
To determine whether additive composition of diffusion-based energies,
$$
E_{4\wedge7}(x) = E_4(x) + E_7(x),
$$
yields valid samples **if and only if** (i) the experts’ data distributions have overlapping support, and (ii) the model architecture encodes inductive biases consistent with **existential, location-agnostic semantics**.

## Problems Addressed

1. **Support Mismatch.**
   Standard MNIST contains exactly one centered digit per image. As a result, the supports of $(p_4(x))$ and $(p_7(x))$ have negligible or empty intersection. In this regime, PoE composition is ill-posed at the data-distribution level, independent of model choice.

2. **Inductive Bias Mismatch.**
   Even when support overlap is introduced via dataset curation, standard or center-biased CNN architectures often encode assumptions of single-object, centrally-located structure. These assumptions conflict with the intended existential semantics (“a digit exists somewhere”), preventing meaningful energy composition.

3. **Energy-Scale and Sampling Instability.**
   Diffusion experts may learn incompatible score magnitudes or under-trained energy landscapes, causing one expert to dominate or producing fragmented, incoherent samples during PoE sampling.

## Experimental Regimes and Expected Outcomes

**Regime A: Standard MNIST (Support Mismatch)**

* Experts trained on single centered digits (4 or 7).
* PoE composition fails due to lack of overlapping support.
* Failure is expected and not a contradiction of PoE theory.

**Regime B: Curated Overlap + Center-Biased Architecture (Inductive Bias Mismatch)**

* Data includes images with multiple digits and clean separation.
* Architecture compresses spatial information or assumes centering.
* PoE composition remains ineffective despite overlapping support.

**Regime C: Curated Overlap + Fully Convolutional Architecture (Expected Success)**

* Data explicitly enforces existential semantics and overlapping support.
* Architecture is fully convolutional and location-agnostic.
* PoE composition produces samples satisfying both concepts (4 ∧ 7), validating the hypothesis under controlled conditions.

## Scientific Contribution

This experiment demonstrates that **PoE failure modes are diagnostic**, not contradictory:

* Failure under Regimes A and B isolates limitations imposed by data support and architectural inductive bias.
* Success under Regime C establishes that diffusion-based energy landscapes can be meaningfully composed when these constraints are satisfied.

The study thus clarifies **when diffusion-based PoE composition is well-posed**, and provides a principled framework for analyzing compositional generative modeling beyond naive empirical demonstrations.

## Investigating SuperDiff for logical AND

**Brief answer:**
Investigating **SuperDiff** for logical AND would be **conceptually informative but not fundamentally different** from PoE-style energy addition for the task you are studying.

**Why:**

* SuperDiff (Skreta et al.) also performs **logical AND by composing densities**, effectively identifying regions where multiple models assign **comparable log-probability** along the diffusion trajectory.
* In energy terms, this corresponds to operating on the **same underlying object**: combinations of log-densities (energies) that define where probability mass concentrates.
* The “equiprobable region” criterion in SuperDiff is **not a different semantic objective**, but a **different parameterization and control mechanism** for navigating the same composite energy landscape.

**What *is* different:**

* SuperDiff explicitly **tracks log densities along the SDE**, giving finer control over *how* and *when* models influence the sampler.
* This can improve **stability and calibration** when experts have mismatched score magnitudes or when support overlap is weak but nonzero.

**Bottom line:**
SuperDiff does **not change the compositional task** (logical AND via intersecting supports); it changes the **numerical and geometric implementation** of that task. Its significance lies in **stability and interpretability**, not in redefining what logical composition means.

**Expectation.**
SuperDiff would yield **qualitatively similar outcomes** to PoE energy addition in this experiment.

**Justification.**
Both approaches implement the **same compositional objective**—identifying regions where both experts assign high probability (low energy). Given that your experiment already enforces **support overlap** and **existential, location-agnostic inductive bias**, SuperDiff’s density-tracking and equiprobable-region construction would mainly improve **sampling stability and calibration**, not change the set of solutions. The logical AND task itself remains unchanged; only the numerical path taken through the energy landscape differs.

## Results: Assessment of the Regime C samples
### 1) Interpretation Of Results

#### Partial Success
* **“Partial success”**: In the grid, there are multiple tiles that clearly contain **both a 4 and a 7** (sometimes separated), which is exactly the intended conjunction behavior under existential semantics. At the same time, many samples are **single-digit**, **fragmented**, or **noisy**.
* The identified failure modes are plausible and visible:

  * **Missing conjunction**: several tiles show only one digit.
  * **Fragmentation / noise**: many samples have speckle and incomplete strokes.
  * **Occasional “other digits”**: I can visually see things resembling **8, 3, 6, 9** in places, not only 4/7.

#### Limitations

* **“Mode imbalance: more 4+7 than 7+4 placements”** is not well-supported from a single grid and is not a meaningful claim unless you define a left/right placement statistic and measure it over many samples.
* **“Longer training alone won’t fix digit collisions or single-digit collapse if experts are mis-calibrated”**: Longer training helps if the bottleneck is **denoiser quality**; it does not help if the bottleneck is **objective mis-specification** (e.g., your experts don’t penalize extra digits).
* Perhaps our setup implicitly treats “only 4s and 7s” as if it were automatically implied by PoE. **It is not.**
  Under our semantics (“there exists a 4” / “there exists a 7”), the composed distribution still allows **extra digits** unless we explicitly constrain them away.


### 2) Current state: partial success consistent with the hypothesis

* We have evidence that, when support overlap + fully-conv inductive bias are present, **PoE can steer sampling toward intersection-like solutions** (many tiles show both digits).
* However, the sample quality suggests the learned score fields are still **imprecise**: reverse dynamics sometimes lands in:

  1. regions satisfying only one expert (single-digit),
  2. “cheap” local minima (fragmented strokes),
  3. unconstrained content (other digits), which is *allowed* under your current semantics.

* The grid is **credible evidence of partial success**: PoE is sometimes finding low-energy configurations that satisfy both expert constraints.
* The next improvements split into two categories:

  1. **Make samples cleaner and conjunction more reliable** → longer training + EMA + better sampler + calibration.
  2. **Make outputs “only 4s and 7s”** → you need **additional constraints/experts**, because existential PoE alone does not forbid extra digits.
  


### 3) Why “only 4s and 7s” is not guaranteed

Right now your experts are closer to:

* ($E_4(x)$): low energy if a 4 exists somewhere (does **not** forbid other digits)
* ($E_7(x)$): low energy if a 7 exists somewhere (does **not** forbid other digits)

PoE gives:
$$
p_{\wedge}(x)\propto p_4(x)p_7(x)
\quad\Leftrightarrow\quad
E_{\wedge}(x)=E_4(x)+E_7(x).
$$

This construction enforces **conjunction of constraints the experts actually represent**.
If neither expert assigns high energy to “contains an 8”, then the conjunction **does not penalize 8s**. So “only 4s and 7s” is a **stronger statement** than “4 ∧ 7 exists”.

To get “only 4s and 7s”, we must add at least one of:

* a **negative constraint**: “no other digits exist”
* a **count constraint**: “exactly one 4 and exactly one 7”
* extra experts (anti-experts) whose energies penalize unwanted content

This is objective/constraint design, not just training longer.


### 4) What is required to improve results (ordered, with rationale)

####Step 0 — Sanity prerequisites (must be true for PoE score addition to be meaningful)

1. **Identical diffusion schedule for both experts** (same betas, same (T), same parameterization ε vs v).
   Otherwise, “ε addition” is not comparable as a proxy for score addition.
2. Confirm you are composing at sampling as:
   $$
   \varepsilon_{\text{poe}}(x_t,t)=w_4\varepsilon_4(x_t,t)+w_7\varepsilon_7(x_t,t)
   $$
   (and not mixing x0-preds or inconsistent parameterizations).

####Step 1 — Improve denoiser quality (fix fragmentation/noise)

This targets the “speckle + incomplete strokes” failure mode.

* Train longer **and** use **EMA**.
  EMA approximates time-averaging of parameters and typically yields a smoother score field, which reduces sampling artifacts.
* Increase capacity modestly (more channels / more ResBlocks).
  Better function class → better approximation to score ($\nabla_{x_t}\log p(x_t)$).
* Use **DDIM / DPM-Solver / Heun** rather than basic ancestral DDPM at low step counts.
  These reduce discretization error in the reverse SDE/ODE and tend to produce cleaner structure for a fixed compute budget.

**Expected improvement:** crisper digits, fewer broken strokes, higher conjunction rate.

####Step 2 — Fix energy/score calibration across experts (fix “single digit collapse”)

Even if both experts are good, PoE can collapse if one expert’s score dominates in magnitude:
$$
|\nabla \log p_4(x_t)|\gg |\nabla \log p_7(x_t)|
\Rightarrow \nabla\log p_{\wedge}\approx \nabla\log p_4
$$
so you get mostly 4-like samples.

Actions:

* **Per-sample ε normalization** (you already discussed this) and a **weight sweep** (($w_4,w_7$)).
* Better: match norms **per timestep** (since score magnitudes vary with t).
* Track conjunction proxy metrics over many samples (not one grid).

**Expected improvement:** fewer samples satisfying only one expert; more stable trade-off.

####Step 3 — Decide what you *actually* want: existential conjunction vs exclusivity

This is the big conceptual fork.

**3A: If your target is only “a 4 exists and a 7 exists”**
Then other digits are not strictly “wrong,” but you might still want to *discourage* them. You can do that via dataset design (no distractors) and stronger experts.

**3B: If your target is “only 4s and 7s appear”**
You must add constraints beyond ($E_4 + E_7$). Three principled options:

1. **Anti-experts for other digits**
   Train experts ($E_k(x)$) for all ($k\neq 4$,7), and compose:
   $$
   E(x)=E_4(x)+E_7(x);-;\lambda\sum_{k\neq 4,7} E_k(x)
   $$
   (negative weight means “repel” those concepts). This is the cleanest PoE/EBM formulation.

2. **A learned “only {4,7}” constraint energy**
   Train a classifier/energy (E_{\text{only47}}(x)) that penalizes any patch containing digits not in {4,7}. Add it as another expert:
   $$
   E(x)=E_4(x)+E_7(x)+\alpha E_{\text{only47}}(x).
   $$

3. **Count-aware constraint** (“exactly one 4 and exactly one 7”)
   Add a differentiable count proxy (e.g., from a sliding-window detector producing soft counts) and penalize deviations:
   $$
   E_{\text{count}}(x)= (c_4(x)-1)^2+(c_7(x)-1)^2.
   $$
   This directly aligns the energy with your desired output distribution.

**Expected improvement:** suppression of “8/3/6/9” artifacts even when denoising is good.

####Step 4 — Measurement (so you can tell which knob matters)

For each run, report:

* Expert quality: ($\mathbb{E}[\text{exists4}(x)]$) for samples from ($p_4$), similarly for ($p_7$).
* Conjunction rate: ( $\mathbb{E}[\min(\text{exists4},\text{exists7})]$ ) on PoE samples.
* Exclusivity rate (if desired): ( $\mathbb{E}[1-\max_{k\notin{4,7}}\text{exists}k$] ).
* Calibration curves vs timestep if you can (or just norm stats per t).

This turns “looks better” into falsifiable progress.

### Formulation A — Existential conjunction (what Regime C currently implements)

#### Target distribution

Let ($x\in[-1,1]^{H\times W}$) be a canvas image. Define the *event*
($\mathcal{A}_k={\text{“a digit }k\text{ occurs at least once in }x”}$).

Your intended target is the conditional distribution
$$
p(x \mid \mathcal{A}_4 \cap \mathcal{A}_7),
$$
i.e., samples from the data manifold that satisfy **existence of both 4 and 7**, with no exclusivity requirement.

####Expert energies and PoE

Train two diffusion experts that model (smoothed) marginals consistent with those events:
$$
p_4(x)\approx p(x\mid \mathcal{A}*4),\qquad
p_7(x)\approx p(x\mid \mathcal{A}*7).
$$
Define the PoE composition
$$
p*{\wedge}(x);\propto;p_4(x),p_7(x)
\quad\Longleftrightarrow\quad
E*{\wedge}(x)=E_4(x)+E_7(x),
$$
where ($E_i(x)=-\log p_i(x)$) up to a constant.

####Diffusion implementation (score/ε composition)

Let ($s_i(x_t,t)\approx \nabla_{x_t}\log p_i(x_t)$) be the expert score. If your models predict ($\varepsilon_i(x_t,t)$), use the standard relation (VP-DDPM):
$$
s_i(x_t,t) \approx -\frac{1}{\sigma_t},\varepsilon_i(x_t,t),
$$
so PoE in score space is:
$$
s_{\wedge}(x_t,t)=s_4(x_t,t)+s_7(x_t,t).
$$
Equivalently in ε-parameterization (same schedule, same (T)):
$$
\varepsilon_{\wedge}(x_t,t)=\varepsilon_4(x_t,t)+\varepsilon_7(x_t,t)
$$
(optionally with calibrated weights and normalization).

####What this guarantees (and what it does not)

* It targets **high density under both experts** ⇒ increased probability of ($\mathcal{A}_4\cap\mathcal{A}_7$).
* It does **not** penalize extra digits. So “only 4 and 7” is **not implied** by this formulation.

### Formulation BFormulation B — Exclusive conjunction (what you need for “only 4s and 7s”)

Now define an exclusivity event:
$$
\mathcal{B}*{47}={\text{all digits in }x \text{ belong to }{4,7}},
$$
and (optionally) a count event:
$$
\mathcal{C}*{1,1}={\text{exactly one 4 and exactly one 7 occur in }x}.
$$

#### Target distribution

For “only 4 and 7 appear (at least once each)”:
$$
p(x \mid \mathcal{A}_4 \cap \mathcal{A}*7 \cap \mathcal{B}*{47}).
$$
For “exactly one 4 and one 7 (and nothing else)”:
$$
p(x \mid \mathcal{A}*4 \cap \mathcal{A}*7 \cap \mathcal{B}*{47}\cap \mathcal{C}*{1,1}).
$$

####Add the missing constraint as an additional expert

You need **at least one extra energy term** that penalizes non-({4,7}) content.

#### Option 1: Anti-experts for other digits (principled PoE/EBM)

Train experts for all digits ($k\in{0,\dots,9}$) on existential semantics. Compose:
$$
E(x)=E_4(x)+E_7(x);-;\lambda\sum_{k\notin{4,7}}E_k(x),
$$
i.e.
$$
\log p(x)=\log p_4(x)+\log p_7(x)-\lambda\sum_{k\notin{4,7}}\log p_k(x)+\text{const}.
$$
In score space:
$$
s(x_t,t)=s_4(x_t,t)+s_7(x_t,t)-\lambda\sum_{k\notin{4,7}}s_k(x_t,t).
$$
This explicitly **repels** other digits.

#### Option 2: Single “only-47” constraint expert (lighter-weight)

Train a classifier/detector ($q_{\text{only47}}(x)\approx \mathbb{P}(\mathcal{B}*{47}\mid x)$). Define a constraint energy:
$$
E*{\text{only47}}(x)=-\log q_{\text{only47}}(x).
$$
Then compose
$$
E(x)=E_4(x)+E_7(x)+\alpha E_{\text{only47}}(x).
$$
In practice, implement ($E_{\text{only47}}$) as a differentiable guidance term (classifier guidance), adding its gradient to the diffusion score:
$$
s(x_t,t)=s_4(x_t,t)+s_7(x_t,t)+\alpha \nabla_{x_t}\log q_{\text{only47}}(x_t).
$$

#### Option B3: Count-aware constraint (for “exactly one each”)

Let a sliding-window detector produce soft counts ($c_4(x),c_7(x)$). Define
$$
E_{\text{count}}(x)=(c_4(x)-1)^2+(c_7(x)-1)^2.
$$
Compose
$$
E(x)=E_4(x)+E_7(x)+\beta E_{\text{count}}(x)+\alpha E_{\text{only47}}(x).
$$

