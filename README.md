<h1 align="center">The Superposition of Diffusion Models Using the Itô Density Estimator</h1>

<p align="center">
<a href="https://arxiv.org/abs/2412.17762"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
<a href="https://colab.research.google.com/drive/1iCEiQUMXmQREjT6pUYQ6QOw1_0EAqa82"><img src="https://img.shields.io/badge/Colab-e37e3d.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Jupyter"/></a>
<a href="https://huggingface.co/superdiff/"><img src="https://img.shields.io/badge/HuggingFace-1f27ca.svg?style=for-the-badge&logo=HuggingFace&logoColor=yellow" alt="HF"/></a>
</p>

The principled method for efficiently combining multiple pre-trained diffusion models solely during inference! We provide a new approach for estimating density without touching the divergence. This gives us the control to easily interpolate concepts (logical AND) or mix densities (logical OR), allowing us to create one-of-a-kind generations!

<p align="center">
<img src="assets/SD_examples.gif" alt="Animation of the examples of SuperDiff for StableDiffusion"/>
</p>

</details>

## Install dependencies

For Stable Diffusion examples, see [Installing Dependencies for CIFAR and SD](/applications/images/README.md)

For Protein examples, see [Installing Dependencies for Protein Models](/applications/proteins/README.md)

## Using Code & Examples

We outline the high-level organization of the repo in the **_project tree_** and provide links to specific examples, notebooks, and experiments in the **_introduction_**, **_CIFAR_**, **_Stable Diffusion (SD)_**, and **_Proteins_** sections. 

#### Project Tree

```
├── applications
│   ├── images
│           - directory for reproducing SD experiments
│   └── proteins
│           - directory for reproducing protein experiments
├── assets
│       - folder with images
├── cifar
│       - directory for reproducing CIFAR experiments
├── LICENSE
├── notebooks
│       - educational examples and notebooks for SD
└── README.md
```

#### Introduction and Educational Notebooks

**Diffusion ([diffusion_edu.ipynb](/notebooks/diffusion_edu.ipynb)):** for an introduction to diffusion models and a basic example of training and sampling.

**Superposition ([superposition_edu.ipynb](/notebooks/superposition_edu.ipynb)):** for an introduction to combining diffusion models and reproducing  Figure 2.


#### CIFAR

**Train:** example for training a single model on CIFAR-10
```
python cifar/main.py --config cifar/configs/sm/cifar/vpsde.py --workdir $PWD/cifar/chkpt/ --mode 'train'
```

**Eval:** example for evaluating a single model on CIFAR-10
```
python cifar/main.py --config cifar/configs/sm/cifar/vpsde.py --workdir $PWD/cifar/chkpt/ --mode 'eval_fid'
```

#### Stable Diffusion (SD)

**Superposition AND ([superposition_AND.ipynb](/notebooks/superposition_AND.ipynb)):** notebook consisting of examples for generating images and interpolating concepts using SuperDiff (AND) with SD.

**Superposition OR ([superposition_OR.ipynb](/notebooks/superposition_OR.ipynb)):** notebook consisting of examples for of generating images using SuperDiff (OR) with SD.

**[SD Experiments](/applications/images/README.md):** for an example of how to generate images using SuperDiff with SD and reproducing the SD experiments.

#### Proteins

**[Protein Experiments](/applications/proteins/README.md):** for an example of how to generate proteins with SuperDiff and reproducing the protein experiments.

Here is a guide describing the necessary steps to prepare and run the CIFAR-10 and split-MNIST (0-4 vs 5-9) experiments.

```markdown
# Experiment Preparation and Execution Guide

This guide outlines the steps to set up the codebase, apply necessary fixes, and launch experiments for both **CIFAR-10** and **Split-MNIST**.

## 1. Environment Setup

Ensure your environment is set up with the necessary dependencies (JAX, TensorFlow, Flax, etc.).

```bash
# Example setup (adjust based on your cluster/local machine)
conda create -n jax115 python=3.9
conda activate jax115
pip install -r requirements.txt

```

## 2. Codebase Preparation (Crucial for MNIST)

The original codebase requires specific modifications to support split-MNIST training (training on digits <5 and >5 separately) and correct evaluation.

### A. Modify `cifar/datasets.py`

Update `get_dataset` to handle custom split names defined in configs.

1. **Locate** the `MNIST` block in `get_dataset`.
2. **Replace** the hardcoded `train_split` and `eval_split` with dynamic lookups:
```python
if config.data.dataset == 'MNIST':
  dataset_builder = tfds.builder('mnist')
  # Allow config to override splits (e.g., 'train<5')
  train_split_name = getattr(config.data, 'train_split', 'train')
  eval_split_name = getattr(config.data, 'eval_split', 'test')
  # ... rest of the resizing logic

```



### B. Modify `cifar/evaluation.py`

Update `load_dataset_stats` to support MNIST and split-specific statistic files.

1. **Remove** the check that forces `config.data.dataset == 'CIFAR10'`.
2. **Add logic** to append `_low` or `_high` to the filename based on the split:
```python
def load_dataset_stats(config, eval=False):
  suffix = 'test' if eval else 'train'
  dataset_name = config.data.dataset.lower()

  # Detect split from config
  train_split = getattr(config.data, 'train_split', '')
  if '<5' in train_split:
    dataset_name += '_low'
  elif '>5' in train_split:
    dataset_name += '_high'

  filename = f'assets/stats/{dataset_name}_{suffix}_stats.npz'
  # ... load and return np.load(fin)

```



### C. Modify `cifar/run_lib.py`

Update `fid_stats` to save statistics with the correct naming convention (`_low` / `_high`) and handle grayscale images.

1. **Match naming logic** used in `evaluation.py` (checking `<5` or `>5`).
2. **Handle Channels**: MNIST is grayscale (1 channel), but Inception expects 3.
```python
# Inside fid_stats loop:
if batch.shape[-1] == 1:
  batch = np.repeat(batch, 3, axis=-1)

```



---

## 3. Running Split-MNIST Experiments

You need to train two separate models: one for digits 0-4 ("Low") and one for digits 5-9 ("High").

### Step 1: Create Configurations

Create two new config files in `cifar/configs/`:

* **`mnist_low.py`**: Set `config.data.train_split = 'train<5'` and `config.data.eval_split = 'test<5'`.
* **`mnist_high.py`**: Set `config.data.train_split = 'train>5'` and `config.data.eval_split = 'test>5'`.

### Step 2: Generate FID Statistics (Run Once)

Before training, generate the reference statistics for evaluation. This prevents crashes during the first evaluation step.

**For Digits 0-4:**

```bash
./launch_train.sh \
  --config configs/mnist_low.py \
  --workdir . \
  --mode fid_stats \
  --name mnist_low_stats

```

**For Digits 5-9:**

```bash
./launch_train.sh \
  --config configs/mnist_high.py \
  --workdir . \
  --mode fid_stats \
  --name mnist_high_stats

```

### Step 3: Launch Training

Now launch the training jobs. Use unique working directories.

**Train Low (0-4):**

```bash
./launch_train.sh \
  --config configs/mnist_low.py \
  --workdir exp_output/mnist_low_run1 \
  --mode train \
  --name mnist_low_train

```

**Train High (5-9):**

```bash
./launch_train.sh \
  --config configs/mnist_high.py \
  --workdir exp_output/mnist_high_run1 \
  --mode train \
  --name mnist_high_train

```

---

## 4. Running CIFAR-10 Experiments

CIFAR-10 experiments use the standard `vpsde` configurations.

### Step 1: Generate Statistics (If missing)

Ensure `assets/stats/cifar10_train_stats.npz` and `cifar10_test_stats.npz` exist. If not:

```bash
./launch_train.sh \
  --config configs/sm/cifar/vpsde.py \
  --workdir . \
  --mode fid_stats \
  --name cifar_stats

```

### Step 2: Launch Training

Run the standard Variance Preserving SDE (VPSDE) training.

```bash
./launch_train.sh \
  --config configs/sm/cifar/vpsde.py \
  --workdir exp_output/cifar_vpsde_run1 \
  --mode train \
  --name cifar_vpsde_train

```

## Summary Checklist

* [x] **Code**: `datasets.py` updated to read split from config?
* [x] **Code**: `evaluation.py` and `run_lib.py` updated to handle `_low`/`_high` filenames?
* [x] **Configs**: `mnist_low.py` and `mnist_high.py` created?
* [x] **Stats**: Ran `fid_stats` for MNIST Low, MNIST High, and CIFAR?
* [x] **Train**: Launch training jobs with distinct `workdir` paths.


## Citation

<div align="left">
  
If you find this code useful in your research, please cite the following paper (expand for BibTeX):

<details>
<summary>
M. Skreta*, L. Atanackovic*, A.J. Bose, A. Tong, K. Neklyudov. The Superposition of Diffusion Models Using the Itô Density Estimator, 2024.
</summary>

```bibtex
@article{skreta2024superposition,
  title={The Superposition of Diffusion Models Using the It$\backslash$\^{} o Density Estimator},
  author={Skreta, Marta and Atanackovic, Lazar and Bose, Avishek Joey and Tong, Alexander and Neklyudov, Kirill},
  journal={arXiv preprint arXiv:2412.17762},
  year={2024}
}
```
