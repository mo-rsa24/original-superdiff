import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from absl import app, flags

from cifar.dataset_regimes import PadTo48, filter_digit_subset, TwoDigitMNISTCanvasClean, TwoDigitMNISTCanvasCleanPlus
from cifar.dynamics.diffusion import DiffusionSchedule, DiffusionConfig, ddim_sample, ddim_sample_poe
from cifar.models.experts import CenterBiasedExpert, FullyConvExpertBigger

FLAGS = flags.FLAGS
flags.DEFINE_enum("regime", "C", ["A", "B", "C"], "Experimental Regime (A, B, or C)")
flags.DEFINE_enum("mode", "dataset", ["dataset", "model"], "Visualize 'dataset' samples or 'model' generations")
flags.DEFINE_string("save_dir", "assets/results", "Directory to save visualizations")
flags.DEFINE_string("weights_4", None, "Path to Expert 4 weights (for mode=model)")
flags.DEFINE_string("weights_7", None, "Path to Expert 7 weights (for mode=model)")
flags.DEFINE_integer("num_samples", 16, "Number of samples to visualize")
flags.DEFINE_integer("seed", 42, "Random seed")


def save_image_grid(tensor, filename, nrow=4, title=None):
    """Saves a grid of images."""
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)

    grid = make_grid(tensor, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(grid_np, vmin=0, vmax=1)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filename}")


def get_dataset_loaders(regime, mnist_base, batch_size):
    """Setup datasets based on Regime logic."""
    if regime == "A":
        # Regime A: Centered single digits (Support Mismatch)
        ds4 = PadTo48(filter_digit_subset(mnist_base, 4))
        ds7 = PadTo48(filter_digit_subset(mnist_base, 7))
        # No ground-truth "both" dataset for Regime A, but we can visualize a dummy overlap
        ds_both = None

    elif regime == "B":
        # Regime B: Clean canvas but center-biased margins (Inductive Bias Mismatch)
        ds4 = TwoDigitMNISTCanvasClean(mnist_base, mode="exists4", digit_size_range=(20, 22), min_margin=14,
                                       seed=FLAGS.seed)
        ds7 = TwoDigitMNISTCanvasClean(mnist_base, mode="exists7", digit_size_range=(20, 22), min_margin=14,
                                       seed=FLAGS.seed + 1)
        ds_both = TwoDigitMNISTCanvasClean(mnist_base, mode="both47", digit_size_range=(20, 22), min_margin=4,
                                           seed=FLAGS.seed + 2)

    elif regime == "C":
        # Regime C: Random placement with distractors (Matched Bias)
        ds4 = TwoDigitMNISTCanvasCleanPlus(
            mnist_base, mode="exists4", forbid_digit=4, digit_size_range=(18, 22),
            min_margin=4, seed=FLAGS.seed, p_extra=0.4,
            target_overlap_digit=7, target_overlap_prob=0.35)
        ds7 = TwoDigitMNISTCanvasCleanPlus(
            mnist_base, mode="exists7", forbid_digit=7, digit_size_range=(18, 22),
            min_margin=4, seed=FLAGS.seed + 1, p_extra=0.4,
            target_overlap_digit=4, target_overlap_prob=0.35)
        ds_both = TwoDigitMNISTCanvasCleanPlus(mnist_base, mode="both47", digit_size_range=(18, 22), min_margin=4,
                                               seed=FLAGS.seed + 2)

    return ds4, ds7, ds_both


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Regime {FLAGS.regime} in {FLAGS.mode} mode on {device}...")

    # Load MNIST
    tfm = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    if FLAGS.mode == "dataset":
        ds4, ds7, ds_both = get_dataset_loaders(FLAGS.regime, mnist_train, FLAGS.num_samples)

        # Visualize 4s
        x4, _ = next(iter(DataLoader(ds4, batch_size=FLAGS.num_samples, shuffle=True)))
        save_image_grid(x4, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_data_4.png"),
                        title=f"Regime {FLAGS.regime}: Expert 4 Data")

        # Visualize 7s
        x7, _ = next(iter(DataLoader(ds7, batch_size=FLAGS.num_samples, shuffle=True)))
        save_image_grid(x7, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_data_7.png"),
                        title=f"Regime {FLAGS.regime}: Expert 7 Data")

        # Visualize Both (if applicable)
        if ds_both:
            xb, _ = next(iter(DataLoader(ds_both, batch_size=FLAGS.num_samples, shuffle=True)))
            save_image_grid(xb, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_data_both.png"),
                            title=f"Regime {FLAGS.regime}: Both (Target)")

    elif FLAGS.mode == "model":
        if not FLAGS.weights_4 or not FLAGS.weights_7:
            raise ValueError("Must provide --weights_4 and --weights_7 for model visualization.")

        # Architecture Config
        if FLAGS.regime == "B":
            ModelCls = CenterBiasedExpert
            base_ch = 64
            n_blocks = 0  # unused
        else:
            ModelCls = FullyConvExpertBigger
            base_ch = 96
            n_blocks = 6

        # Initialize Models
        model4 = ModelCls(base=base_ch, n_blocks=n_blocks).to(device)
        model7 = ModelCls(base=base_ch, n_blocks=n_blocks).to(device)

        # Load Weights
        print(f"Loading weights from {FLAGS.weights_4} and {FLAGS.weights_7}")
        model4.load_state_dict(torch.load(FLAGS.weights_4, map_location=device))
        model7.load_state_dict(torch.load(FLAGS.weights_7, map_location=device))

        schedule = DiffusionSchedule(DiffusionConfig(T=500), device=device)

        # Sample Expert 4
        print("Sampling Expert 4...")
        x4 = ddim_sample(model4, schedule, shape=(FLAGS.num_samples, 1, 48, 48), steps=100, device=device)
        save_image_grid(x4, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_sample_4.png"),
                        title=f"Regime {FLAGS.regime}: Expert 4 Gen")

        # Sample Expert 7
        print("Sampling Expert 7...")
        x7 = ddim_sample(model7, schedule, shape=(FLAGS.num_samples, 1, 48, 48), steps=100, device=device)
        save_image_grid(x7, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_sample_7.png"),
                        title=f"Regime {FLAGS.regime}: Expert 7 Gen")

        # Sample PoE (Product of Experts)
        print("Sampling PoE...")
        x_poe = ddim_sample_poe(model4, model7, schedule, shape=(FLAGS.num_samples, 1, 48, 48), steps=100,
                                device=device)
        save_image_grid(x_poe, os.path.join(FLAGS.save_dir, f"regime_{FLAGS.regime}_sample_poe.png"),
                        title=f"Regime {FLAGS.regime}: PoE Gen")


if __name__ == "__main__":
    app.run(main)