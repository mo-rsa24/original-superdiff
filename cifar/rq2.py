import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from absl import app, flags
from ml_collections import config_flags

from cifar.dataset_regimes import ColoredMNISTAttributes, worker_init_fn
from cifar.dynamics.diffusion import DiffusionConfig, DiffusionSchedule, ddim_sample, ddim_sample_composed
from cifar.models.experts import FullyConvExpertBigger, CenterBiasedExpert
from cifar.regime_evaluation import train_mnist_classifier

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Experiment configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Working directory for checkpoints and samples.")
flags.DEFINE_string("mode", "train", "Mode: train or sample.")
flags.DEFINE_string("expert", "shape", "Expert to train: shape or color.")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_images_grid(x, title, save_path, nrow=8, pad=2):
    x = x.detach().cpu()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)

    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=pad)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()


def build_color_mnist_dataset(config, mode):
    tfm = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return ColoredMNISTAttributes(
        mnist_train,
        mode=mode,
        target_digit=config.target_digit,
        target_color=config.target_color,
        palette=config.color_palette,
        color_jitter=config.color_jitter,
        seed=config.seed,
    )


def build_expert_model(config):
    if config.model_arch == "FullyConvExpertBigger":
        return FullyConvExpertBigger(
            in_ch=config.num_channels,
            base=config.base_channels,
            n_blocks=config.n_blocks,
        )
    if config.model_arch == "CenterBiasedExpert":
        return CenterBiasedExpert(in_ch=config.num_channels, base=config.base_channels)
    raise ValueError(f"Unknown model_arch {config.model_arch}")


def train_expert(config, workdir, expert_type, device):
    dataset = build_color_mnist_dataset(config, mode="shape" if expert_type == "shape" else "color")
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    schedule = DiffusionSchedule(DiffusionConfig(**config.diffusion), device=device)
    model = build_expert_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = nn.MSELoss()

    step = 0
    start_time = time.time()
    for epoch in range(1000000):
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(0, schedule.cfg.T, (x.size(0),), device=device)
            xt, noise = schedule.q_sample(x, t)
            pred = model(xt, t.float())
            loss = criterion(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % config.train.log_every == 0:
                elapsed = time.time() - start_time
                print(f"[{expert_type}] step {step} | loss {loss.item():.4f} | {elapsed:.1f}s")

            if step % config.train.sample_every == 0 and step > 0:
                sample = ddim_sample(model, schedule, (config.sampling.num_samples, config.num_channels,
                                                     config.image_size, config.image_size),
                                     steps=config.sampling.steps, eta=config.sampling.eta, seed=config.seed, device=device)
                save_images_grid(
                    sample,
                    title=f"{expert_type}_sample_step_{step}",
                    save_path=os.path.join(workdir, "samples", f"{expert_type}_step_{step}.png"),
                )

            step += 1
            if step >= config.train.steps:
                ckpt_path = os.path.join(workdir, f"{expert_type}_expert.pt")
                _ensure_dir(workdir)
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved {expert_type} checkpoint to {ckpt_path}")
                return


def load_expert(config, workdir, expert_type, device):
    model = build_expert_model(config).to(device)
    ckpt_path = os.path.join(workdir, f"{expert_type}_expert.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def to_grayscale(x):
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return x.mean(dim=1, keepdim=True).clamp(0, 1)


def shape_accuracy(samples, clf, target_digit, device):
    x_gray = to_grayscale(samples).to(device)
    logits = clf(x_gray)
    preds = torch.argmax(logits, dim=1)
    return float((preds == target_digit).float().mean().item())


def color_accuracy(samples, target_color):
    if samples.min() < 0:
        samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0, 1)

    mask = samples.mean(dim=1, keepdim=True) > 0.2
    masked = samples * mask.float()
    mean_rgb = masked.flatten(2).mean(dim=2)

    target_channels = {
        "red": [0],
        "green": [1],
        "blue": [2],
        "yellow": [0, 1],
        "cyan": [1, 2],
        "magenta": [0, 2],
    }
    channels = target_channels.get(target_color, [2])
    target_score = mean_rgb[:, channels].mean(dim=1)
    other_channels = [i for i in range(3) if i not in channels]
    other_score = mean_rgb[:, other_channels].mean(dim=1)
    return float((target_score > other_score).float().mean().item())


def sample_regimes(config, workdir, device):
    shape_model = load_expert(config, workdir, "shape", device)
    color_model = load_expert(config, workdir, "color", device)
    schedule = DiffusionSchedule(DiffusionConfig(**config.diffusion), device=device)

    tfm = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    clf = train_mnist_classifier(mnist_train, device=device, epochs=2)

    results = {}
    shape = (config.sampling.num_samples, config.num_channels, config.image_size, config.image_size)

    for regime in config.regimes:
        if regime == "shape_only":
            samples = ddim_sample(shape_model, schedule, shape, steps=config.sampling.steps,
                                  eta=config.sampling.eta, seed=config.seed, device=device)
        elif regime == "color_only":
            samples = ddim_sample(color_model, schedule, shape, steps=config.sampling.steps,
                                  eta=config.sampling.eta, seed=config.seed, device=device)
        elif regime == "shape_and_color":
            samples = ddim_sample_composed(
                [shape_model, color_model],
                schedule,
                shape,
                steps=config.sampling.steps,
                eta=config.sampling.eta,
                normalize_eps=config.sampling.normalize_eps,
                renormalize_sum=config.sampling.renormalize_sum,
                method="poe",
                seed=config.seed,
                device=device,
            )
        else:
            raise ValueError(f"Unknown regime: {regime}")

        save_images_grid(
            samples,
            title=regime,
            save_path=os.path.join(workdir, "samples", f"{regime}.png"),
        )

        shape_acc = shape_accuracy(samples, clf, config.target_digit, device)
        color_acc = color_accuracy(samples, config.target_color)
        results[regime] = {
            "shape_accuracy": shape_acc,
            "color_accuracy": color_acc,
            "conjunction_accuracy": float(shape_acc * color_acc),
        }

    if config.enable_superdiff_and:
        samples, eps_stats = ddim_sample_composed(
            [shape_model, color_model],
            schedule,
            shape,
            steps=config.sampling.steps,
            eta=config.sampling.eta,
            normalize_eps=config.sampling.normalize_eps,
            renormalize_sum=config.sampling.renormalize_sum,
            method="superdiff_and",
            seed=config.seed,
            device=device,
            return_eps_stats=True,
        )
        save_images_grid(
            samples,
            title="superdiff_and",
            save_path=os.path.join(workdir, "samples", "superdiff_and.png"),
        )
        sd_shape = shape_accuracy(samples, clf, config.target_digit, device)
        sd_color = color_accuracy(samples, config.target_color)
        results["superdiff_and"] = {
            "shape_accuracy": sd_shape,
            "color_accuracy": sd_color,
            "conjunction_accuracy": float(sd_shape * sd_color),
        }
        eps_path = os.path.join(workdir, "samples", "superdiff_and_eps_stats.json")
        with open(eps_path, "w", encoding="utf-8") as f:
            json.dump(eps_stats, f, indent=2)

    metrics_path = os.path.join(workdir, "samples", "regime_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir or "./attribute_composition_runs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if FLAGS.mode == "train":
        if FLAGS.expert not in {"shape", "color"}:
            raise ValueError("expert must be 'shape' or 'color' for training.")
        train_expert(config, workdir, FLAGS.expert, device)
    elif FLAGS.mode == "sample":
        sample_regimes(config, workdir, device)
    else:
        raise ValueError("mode must be 'train' or 'sample'.")


if __name__ == "__main__":
    app.run(main)