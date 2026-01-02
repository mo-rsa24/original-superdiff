import argparse
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from ml_collections import config_flags
from absl import app, flags
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from cifar.dynamics.diffusion import DiffusionSchedule, DiffusionConfig, ddim_sample_poe

from cifar.dataset_regimes import filter_digit_subset, PadTo48, TwoDigitMNISTCanvasClean, TwoDigitMNISTCanvasCleanPlus, \
    worker_init_fn
from cifar.models.experts import CenterBiasedExpert, FullyConvExpertBigger
from cifar.regime_evaluation import train_mnist_classifier, eval_existential
from cifar.regime_utils import train_expert_steps

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", "train", "Mode: train or eval")


def save_images_grid(x, title, save_path, nrow=8, pad=2):
    """
    Saves a grid of images to a file.
    Expects x in format (B, C, H, W).
    """
    x = x.detach().cpu()
    # Normalize to [0, 1] if in [-1, 1]
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)

    # make_grid produces (C, H, W)
    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=pad)
    # Permute to (H, W, C) for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    # If the image is grayscale (C=1), map to gray. make_grid usually outputs 3 channels,
    # but we handle the case just to be safe or if user configured otherwise.
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np[:, :, 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(grid_np, vmin=0, vmax=1)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load MNIST Base
    tfm = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    cfg = FLAGS.config
    schedule = DiffusionSchedule(DiffusionConfig(T=500), device=device)

    # -- Dataset Setup --
    if cfg.regime == "A":
        ds4 = PadTo48(filter_digit_subset(mnist_train, 4))
        ds7 = PadTo48(filter_digit_subset(mnist_train, 7))
    elif cfg.regime == "B":
        ds4 = TwoDigitMNISTCanvasClean(mnist_train, mode="exists4", digit_size_range=cfg.digit_size_range,
                                       min_margin=cfg.min_margin)
        ds7 = TwoDigitMNISTCanvasClean(mnist_train, mode="exists7", digit_size_range=cfg.digit_size_range,
                                       min_margin=cfg.min_margin)
    elif cfg.regime == "C":
        ds4 = TwoDigitMNISTCanvasCleanPlus(mnist_train, mode="exists4", forbid_digit=4,
                                           digit_size_range=cfg.digit_size_range, min_margin=cfg.min_margin)
        ds7 = TwoDigitMNISTCanvasCleanPlus(mnist_train, mode="exists7", forbid_digit=7,
                                           digit_size_range=cfg.digit_size_range, min_margin=cfg.min_margin)

    dl4 = DataLoader(ds4, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
                     worker_init_fn=worker_init_fn if cfg.regime != "A" else None)
    dl7 = DataLoader(ds7, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
                     worker_init_fn=worker_init_fn if cfg.regime != "A" else None)

    # -- Model Setup --
    ModelCls = CenterBiasedExpert if cfg.model_arch == "CenterBiasedExpert" else FullyConvExpertBigger
    # Note: CenterBiasedExpert ignores n_blocks
    model4 = ModelCls(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)
    model7 = ModelCls(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)

    if FLAGS.mode == "train":
        print(f"Training Regime {cfg.regime} (Expert 4)...")
        m4, ema4, _ = train_expert_steps(model4, dl4, schedule, device, max_steps=cfg.train_steps)
        torch.save(ema4.shadow, os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth"))

        print(f"Training Regime {cfg.regime} (Expert 7)...")
        m7, ema7, _ = train_expert_steps(model7, dl7, schedule, device, max_steps=cfg.train_steps)
        torch.save(ema7.shadow, os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth"))

    elif FLAGS.mode == "eval":
        # Load weights
        model4.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth")))
        model7.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth")))

        # Train classifier for eval
        print("Training Evaluation Classifier...")
        clf = train_mnist_classifier(mnist_train, device)

        # Sample PoE
        print("Sampling PoE...")
        x_poe = ddim_sample_poe(model4, model7, schedule, shape=(64, 1, 48, 48), steps=100, device=device)

        # Evaluate
        metrics = eval_existential(x_poe, clf, device)
        print(f"Regime {cfg.regime} Results:", metrics)


if __name__ == "__main__":
    app.run(main)