## Running the scripts

### **Important Setup Note**

For these scripts to run correctly in PyCharm, you must set the **Working directory** in your Run Configuration to the root folder of the project (the parent of the `cifar` folder), typically:
`.../original-superdiff-regime-c-idea-1-poe-overlap-calibration/`

---

### **1. Script: `cifar/rq1.py` (Training & Evaluation)**

These configurations train the experts. Change `--mode=train` to `--mode=eval` to run evaluation on existing checkpoints.

**MNIST Regimes (Full Training)**

* **Regime A:**
```text
--config=cifar/configs/sm/mnist/regime_a.py --workdir=runs/mnist/regime_a --mode=train

```


* **Regime B:**
```text
--config=cifar/configs/sm/mnist/regime_b.py --workdir=runs/mnist/regime_b --mode=train

```


* **Regime C:**
```text
--config=cifar/configs/sm/mnist/regime_c.py --workdir=runs/mnist/regime_c --mode=train

```



**Sanity Checks (Fast/Debug Training)**

* **Sanity Regime A:**
```text
--config=cifar/configs/sm/sanity/regime_a.py --workdir=runs/sanity/regime_a --mode=train

```


* **Sanity Regime B:**
```text
--config=cifar/configs/sm/sanity/regime_b.py --workdir=runs/sanity/regime_b --mode=train

```


* **Sanity Regime C:**
```text
--config=cifar/configs/sm/sanity/regime_c.py --workdir=runs/sanity/regime_c --mode=train

```



---

### **2. Script: `cifar/visualize_rq1.py` (Visualization)**

These configurations visualize the datasets or the outputs of your trained models.

**Mode: Dataset (Visualize Inputs)**

* **Regime A Data:**
```text
--regime=A --mode=dataset --save_dir=assets/visualizations/regime_a

```


* **Regime B Data:**
```text
--regime=B --mode=dataset --save_dir=assets/visualizations/regime_b

```


* **Regime C Data:**
```text
--regime=C --mode=dataset --save_dir=assets/visualizations/regime_c

```



**Mode: Model (Visualize Generations)**
*Note: These assume you have trained the models using the `rq1.py` parameters above. The paths point to where `rq1.py` saves checkpoints by default.*

* **Regime A Model (MNIST):**
```text
--regime=A --mode=model --weights_4=runs/mnist/regime_a/regime_A_expert4.pth --weights_7=runs/mnist/regime_a/regime_A_expert7.pth --save_dir=assets/results/regime_a

```


* **Regime B Model (MNIST):**
```text
--regime=B --mode=model --weights_4=runs/mnist/regime_b/regime_B_expert4.pth --weights_7=runs/mnist/regime_b/regime_B_expert7.pth --save_dir=assets/results/regime_b

```


* **Regime C Model (MNIST):**
```text
--regime=C --mode=model --weights_4=runs/mnist/regime_c/regime_C_expert4.pth --weights_7=runs/mnist/regime_c/regime_C_expert7.pth --save_dir=assets/results/regime_c

```
## Recommended Python version

**Python 3.11**
It’s the safest “middle” for the pinned deep learning stack you have (TF 2.19 / torch 2.5.1 / jax 0.5.3) and is widely supported by wheels on Linux.

---

## System prerequisites (WSL2 Ubuntu)

Most of your stack installs from wheels, but these help avoid annoying build failures (and are generally useful):

```bash
sudo apt update
sudo apt install -y build-essential git curl pkg-config \
  libglib2.0-0 libsm6 libxext6 libxrender1 \
  libstdc++6
```

*(The `libsm6/libx*` bits help with image/GUI backends sometimes used by matplotlib/opencv-like stacks.)*

---

## Clean micromamba environment creation

### 1) Create & activate an env

```bash
micromamba create -n cvgen -c conda-forge python=3.11 pip -y
micromamba activate cvgen
```

### 2) Install “conda-friendly” core libs first (recommended)

This reduces pip resolver pain and speeds things up:

```bash
micromamba install -c conda-forge -y \
  numpy scipy pandas matplotlib seaborn scikit-learn numba \
  pillow imageio pyarrow h5py \
  jupyterlab ipykernel ipywidgets \
  pyyaml requests tqdm rich \
  pyside6
```

> Why this helps: these packages commonly have compiled components; conda-forge handles binaries cleanly.

---

## Install everything from your TXT (pip)

Assuming your file is at `/mnt/data/requirements_clean.txt` :

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r /mnt/data/requirements_clean.txt
```

### If you’re on an NVIDIA GPU in WSL2

Your requirements include CUDA-12 related wheels (e.g., `nvidia-cublas-cu12`, `jax-cuda12-plugin`, etc.) , so pip will pull GPU-enabled bits. Make sure your **Windows** side has a WSL2-compatible NVIDIA driver installed.

---

## Two important notes about your requirements file

1. **Some packages are unpinned** (e.g., `numpy`, `scipy`, `traitlets`, etc.) 
   Pip will choose latest versions that satisfy constraints. That usually works, but if you want full reproducibility, pin them later by freezing after a successful install:

   ```bash
   python -m pip freeze > requirements_locked.txt
   ```

2. **Mixing conda + pip**
   The approach above is the least painful: install compiled “basics” with conda-forge, then install the rest via pip. Avoid re-installing torch/tf/jax via conda if you’re using the pinned pip wheels in the file.

---

## Quick verification commands

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "import tensorflow as tf; print('tf', tf.__version__)"
python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices())"
```

If you paste the error output you get (if any), I’ll tell you exactly which dependency is conflicting and the smallest fix (usually one or two pins).
