import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import copy
import time
import wandb
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from ml_collections import config_flags
from absl import app, flags

from cifar.dynamics.diffusion import DiffusionSchedule, DiffusionConfig, ddim_sample_poe
from cifar.dataset_regimes import filter_digit_subset, PadTo48, TwoDigitMNISTCanvasClean, TwoDigitMNISTCanvasCleanPlus, \
    worker_init_fn
from cifar.models.experts import CenterBiasedExpert, FullyConvExpertBigger
from cifar.regime_evaluation import train_mnist_classifier, eval_existential

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", "train", "Mode: train or eval")
flags.DEFINE_boolean("use_wandb", True, "Whether to use Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "superdiff-rq1", "WandB Project Name")
flags.DEFINE_string("wandb_entity", None, "WandB Entity (User/Team)")
flags.DEFINE_integer("sample_every", 5000, "How often to sample validation images during training.")


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.to(next(model.parameters()).device)
        self.shadow.eval()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow.state_dict()[name].mul_(self.decay).add_(param.data, alpha=(1.0 - self.decay))


def save_images_grid(x, title, save_path, nrow=8, pad=2, log_wandb=False, step=None):
    """
    Saves a grid of images to a file and optionally logs to W&B.
    """
    x = x.detach().cpu()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)

    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=pad)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np[:, :, 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(grid_np, vmin=0, vmax=1)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    if log_wandb and wandb.run is not None:
        wandb.log({title: wandb.Image(plt, caption=f"{title} (Step {step})")})

    plt.close()


@torch.no_grad()
def ddim_sample_single(model, schedule, shape, device, steps=50):
    """
    Samples from a single expert model using DDIM.
    Useful for checking progress of Expert 4 or Expert 7 individually.
    """
    b = shape[0]
    img = torch.randn(shape, device=device)

    # Simple linear spacing of timesteps
    times = torch.linspace(schedule.cfg.T - 1, 0, steps=steps).long().to(device)

    for i in range(len(times)):
        t = times[i]
        t_prev = times[i + 1] if i < len(times) - 1 else -1

        t_batch = torch.full((b,), t, device=device, dtype=torch.long)

        # Get alphas
        alpha_bar = schedule.sqrt_alpha_bar[t]
        alpha_bar_prev = schedule.sqrt_alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # Predict noise
        noise_pred = model(img, t_batch.float())

        # DDIM Step (Equation 12 in DDIM paper)
        # 1. Predict x0
        pred_x0 = (img - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # 2. Point to xt_prev
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
        img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

    return img


def train_expert_wandb(model, dataloader, schedule, device, max_steps, name="expert", log_every=50, sample_every=5000):
    """
    Training loop with periodic sampling.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    ema = EMA(model, decay=0.999)
    criterion = nn.MSELoss()

    iter_dl = iter(dataloader)
    step = 0
    start_time = time.time()

    print(f"Starting training for {name}...")

    while step < max_steps:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dataloader)
            batch = next(iter_dl)

        x, _ = batch
        x = x.to(device)
        #x = x * 2.0 - 1.0  # Normalize [0,1] -> [-1, 1]

        B = x.shape[0]
        t = torch.randint(0, schedule.cfg.T, (B,), device=device).long()
        noise = torch.randn_like(x)

        sqrt_alpha_bar = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

        optimizer.zero_grad()
        noise_pred = model(x_t, t.float())
        loss = criterion(noise_pred, noise)
        loss.backward()

        optimizer.step()
        ema.update()

        step += 1

        # -- Logging --
        if step % log_every == 0:
            current_time = time.time()
            sps = (log_every * B) / (current_time - start_time) if step > log_every else 0
            start_time = current_time

            log_dict = {
                f"{name}/loss": loss.item(),
                f"{name}/step": step,
                f"{name}/samples_per_sec": sps
            }
            print(f"[{name}] Step {step}/{max_steps} | Loss: {loss.item():.4f}")
            if wandb.run is not None:
                wandb.log(log_dict,
                          step=step)  # Align step across models? usually better to use global step or just let wandb handle it.

        # -- Periodic Validation Sampling --
        if step % sample_every == 0 and step > 0:
            print(f"[{name}] Generating validation samples (Online vs EMA) at step {step}...")

            for m_type, m_obj in [("online", model), ("ema", ema.shadow)]:
                m_obj.eval()
                with torch.no_grad():
                    val_samples = ddim_sample_single(m_obj, schedule, shape=(16, 1, 48, 48), device=device, steps=20)

                img_path = os.path.join(FLAGS.workdir, "samples", f"{name}_{m_type}_step_{step}.png")
                save_images_grid(val_samples, f"Val {name} {m_type.upper()}", img_path, nrow=4, log_wandb=True,
                                 step=step)

            model.train()

    return model, ema


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Initialize W&B
    if FLAGS.use_wandb:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            config=FLAGS.config.to_dict(),
            name=f"Regime_{FLAGS.config.regime}_{FLAGS.mode}"
        )

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

    dl4 = DataLoader(ds4, batch_size=cfg.batch_size if 'batch_size' in cfg else 128, shuffle=True, num_workers=2,
                     pin_memory=True)
    dl7 = DataLoader(ds7, batch_size=cfg.batch_size if 'batch_size' in cfg else 128, shuffle=True, num_workers=2,
                     pin_memory=True)

    # -- Model Setup --
    if cfg.model_arch == "CenterBiasedExpert":
        model4 = CenterBiasedExpert(base=cfg.base_channels).to(device)
        model7 = CenterBiasedExpert(base=cfg.base_channels).to(device)
    else:
        model4 = FullyConvExpertBigger(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)
        model7 = FullyConvExpertBigger(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)

    if FLAGS.mode == "train":
        print(f"Training Regime {cfg.regime} (Expert 4)...")
        m4, ema4 = train_expert_wandb(model4, dl4, schedule, device, max_steps=cfg.train_steps, name="expert4",
                                      sample_every=FLAGS.sample_every)
        torch.save(ema4.shadow.state_dict(), os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth"))

        print(f"Training Regime {cfg.regime} (Expert 7)...")
        m7, ema7 = train_expert_wandb(model7, dl7, schedule, device, max_steps=cfg.train_steps, name="expert7",
                                      sample_every=FLAGS.sample_every)
        torch.save(ema7.shadow.state_dict(), os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth"))

        if FLAGS.use_wandb:
            wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth"))
            wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth"))

    if FLAGS.mode == "eval" or FLAGS.mode == "train":
        if FLAGS.mode == "eval":
            print("Loading checkpoints...")
            model4.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth")))
            model7.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth")))
        else:
            model4 = ema4.shadow
            model7 = ema7.shadow

        print("Training Evaluation Classifier...")
        clf = train_mnist_classifier(mnist_train, device)

        print("Sampling PoE (Joint)...")
        x_poe = ddim_sample_poe(model4, model7, schedule, shape=(64, 1, 48, 48), steps=100, device=device)

        img_path = os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_poe_final.png")
        save_images_grid(x_poe, f"Regime {cfg.regime} Final PoE", img_path, log_wandb=FLAGS.use_wandb,
                         step=cfg.train_steps)

        metrics = eval_existential(x_poe, clf, device)
        print(f"Regime {cfg.regime} Results:", metrics)

        if FLAGS.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})


if __name__ == "__main__":
    app.run(main)
