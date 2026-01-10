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
from torchvision import datasets
from torch.utils.data import DataLoader
from ml_collections import config_flags
from absl import app, flags

from cifar.dynamics.diffusion import DiffusionSchedule, DiffusionConfig, ddim_sample_poe
from cifar.dataset_regimes import filter_digit_subset, PadTo48, TwoDigitMNISTCanvasClean, TwoDigitMNISTCanvasCleanPlus, \
    worker_init_fn,  summarize_digit_support
from cifar.models.experts import CenterBiasedExpert, FullyConvExpertBigger, CountConstraintNet
from cifar.regime_evaluation import train_mnist_classifier, eval_existential, eval_only_digits

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

def save_debug_batch(x, workdir, tag="train_batch", step=0, log_wandb=False):
    """Utility to quickly visualize a raw batch."""
    os.makedirs(os.path.join(workdir, "samples"), exist_ok=True)
    save_images_grid(
        x,
        title=f"{tag} (step {step})",
        save_path=os.path.join(workdir, "samples", f"{tag}_step_{step}.png"),
        nrow=8,
        log_wandb=log_wandb,
        step=step,
    )


@torch.no_grad()
def ddim_sample_single(model, schedule, shape, device, steps=50, return_trajectory=False):
    """
    Samples from a single expert model using DDIM.
    Useful for checking progress of Expert 4 or Expert 7 individually.
    """
    b = shape[0]
    img = torch.randn(shape, device=device)

    # Simple linear spacing of timesteps
    times = torch.linspace(schedule.cfg.T - 1, 0, steps=steps).long().to(device)
    trajectory = [] if return_trajectory else None

    for i in range(len(times)):
        t = times[i]
        t_prev = times[i + 1] if i < len(times) - 1 else -1

        t_batch = torch.full((b,), t, device=device, dtype=torch.long)

        # Get alphas
        alpha_bar = schedule.alpha_bar[t]
        alpha_bar_prev = schedule.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # Predict noise
        noise_pred = model(img, t_batch.float())

        # DDIM Step (Equation 12 in DDIM paper)
        # 1. Predict x0
        pred_x0 = (img - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # 2. Point to xt_prev
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
        img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if return_trajectory:
            trajectory.append({
                "t": int(t.item()),
                "x_t": img.detach().cpu(),
                "pred_x0": pred_x0.detach().cpu(),
                "noise_pred": noise_pred.detach().cpu(),
            })

    if return_trajectory:
        return img, trajectory
    return img
@torch.no_grad()
def ddim_sample_poe_debug(model4, model7, schedule, shape, steps=100, eta=0.0,
                          w4=1.0, w7=1.0, normalize_eps=True, renormalize_sum=True,
                          match_eps_norms=False, anti_experts=None, anti_weight=1.0,
                          seed=0, device='cuda', per_timestep_norms=None,
                          return_stats=False, constraint_model=None,
                          constraint_weight=0.0, target_counts=None):
    """
    DDIM sampler for PoE with optional epsilon statistics for debugging.
    Returns samples and (optionally) a dict of eps norms per expert and step.
    """
    torch.manual_seed(seed)
    model4.eval(); model7.eval()
    if anti_experts:
        for model in anti_experts.values():
            model.eval()
    x = torch.randn(shape, device=device)
    T = schedule.cfg.T
    t_seq = torch.linspace(T-1, 0, steps, device=device).long()

    eps_stats = [] if return_stats else None
    if constraint_model is not None:
        constraint_model.eval()
    if target_counts is None:
        target_counts = (1.0, 1.0, 0.0)

    for i in range(len(t_seq)):
        t = t_seq[i].item()
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)

        eps4 = model4(x, tt)
        eps7 = model7(x, tt)
        with torch.no_grad():
            eps4 = model4(x, tt)
            eps7 = model7(x, tt)
            eps_anti = None
            if anti_experts:
                eps_anti = [model(x, tt) for model in anti_experts.values()]
        if normalize_eps:
            eps4 = eps4 / (eps4.std(dim=(1,2,3), keepdim=True) + 1e-6)
            eps7 = eps7 / (eps7.std(dim=(1,2,3), keepdim=True) + 1e-6)
            if eps_anti:
                eps_anti = [eps / (eps.std(dim=(1, 2, 3), keepdim=True) + 1e-6) for eps in eps_anti]
        if match_eps_norms:
            eps4_norm = eps4.flatten(1).norm(dim=1, keepdim=True)
            eps7_norm = eps7.flatten(1).norm(dim=1, keepdim=True)
            target = 0.5 * (eps4_norm + eps7_norm)
            eps4 = eps4 * (target / (eps4_norm + 1e-6)).view(-1, 1, 1, 1)
            eps7 = eps7 * (target / (eps7_norm + 1e-6)).view(-1, 1, 1, 1)
            if eps_anti:
                for idx, eps in enumerate(eps_anti):
                    eps_norm = eps.flatten(1).norm(dim=1, keepdim=True)
                    eps_anti[idx] = eps * (target / (eps_norm + 1e-6)).view(-1, 1, 1, 1)

        eps = w4 * eps4 + w7 * eps7
        if eps_anti:
            eps = eps - anti_weight * torch.stack(eps_anti, dim=0).sum(dim=0)
        if renormalize_sum:
            eps = eps / (eps.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        abar_t = schedule.alpha_bar[t]
        x0 = (x - torch.sqrt(1.0 - abar_t) * eps) / (torch.sqrt(abar_t) + 1e-8)
        if constraint_model is not None and constraint_weight > 0:
            with torch.enable_grad():
                x0_req = x0.detach().requires_grad_(True)
                preds = constraint_model(x0_req)
                target = torch.tensor(target_counts, device=x0_req.device).view(1, 3).repeat(x0_req.size(0), 1)
                loss = torch.mean((preds - target) ** 2)
                grad = torch.autograd.grad(loss, x0_req)[0]
                x0 = (x0_req - constraint_weight * grad).detach()
        if return_stats:
            eps_stats.append({
                "t": t,
                "eps4_norm": float(eps4.flatten(1).norm(dim=1).mean().item()),
                "eps7_norm": float(eps7.flatten(1).norm(dim=1).mean().item()),
                "eps_norm": float(eps.flatten(1).norm(dim=1).mean().item()),
                "eps_anti_norm": float(torch.stack([e.flatten(1).norm(dim=1).mean() for e in eps_anti]).mean().item())
                if eps_anti else 0.0,
            })

        if i == len(t_seq) - 1:
            x = x0
            break

        t_prev = t_seq[i+1].item()
        abar_prev = schedule.alpha_bar[t_prev]
        sigma = 0.0
        if eta > 0:
            sigma = eta * torch.sqrt((1-abar_prev)/(1-abar_t)) * torch.sqrt(1 - abar_t/(abar_prev + 1e-8) + 1e-8)
        z = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = torch.sqrt(abar_prev) * x0 + torch.sqrt(1-abar_prev) * eps + sigma * z

    if return_stats:
        return x, eps_stats
    return x


def _normalize_if_needed(x):
    """Normalize input to [-1, 1] if it is in [0, 1]. Keeps already normalized data intact."""
    min_val = float(x.min().item())
    max_val = float(x.max().item())
    if -1.1 <= min_val <= -0.9 and 0.9 <= max_val <= 1.1:
        return x, (min_val, max_val, "already_-1_1")
    if 0.0 <= min_val and max_val <= 1.0:
        return x * 2.0 - 1.0, (min_val, max_val, "scaled_0_1_to_-1_1")
    return x, (min_val, max_val, "unexpected_range")


def _log_tensor_stats(tag, tensor):
    return {
        f"{tag}/min": float(tensor.min().item()),
        f"{tag}/max": float(tensor.max().item()),
        f"{tag}/mean": float(tensor.mean().item()),
        f"{tag}/std": float(tensor.std().item())
    }

def train_count_constraint(model, dataloader, device, epochs=3, lr=1e-3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            targets = torch.stack([y["count4"], y["count7"], y["count_other"]], dim=1).to(device)
            preds = model(x)
            loss = loss_fn(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"count constraint epoch {ep+1}/{epochs} done")
    return model

def train_expert_wandb(model, dataloader, schedule, device, max_steps, name="expert",
                       log_every=50, sample_every=5000, ema_decay=0.999, use_ema=True):
    """
    Training loop with periodic sampling.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    criterion = nn.MSELoss()

    iter_dl = iter(dataloader)
    step = 0
    start_time = time.time()

    print(f"Starting training for {name}...")
    if hasattr(model, "describe"):
        desc = model.describe()
        print(f"[{name}] architecture: {desc}")
        if wandb.run is not None:
            wandb.log({f"{name}/arch_{k}": v for k, v in desc.items()}, step=0)

    while step < max_steps:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dataloader)
            batch = next(iter_dl)

        x, _ = batch
        x = x.to(device)
        x, x_range_info = _normalize_if_needed(x)

        B = x.shape[0]
        t = torch.randint(0, schedule.cfg.T, (B,), device=device).long()
        noise = torch.randn_like(x)

        sqrt_alpha_bar = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

        optimizer.zero_grad()
        noise_pred = model(x_t, t.float())
        loss = criterion(noise_pred, noise)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / (sqrt_alpha_bar + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        loss.backward()

        optimizer.step()
        if ema is not None:
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
                f"{name}/samples_per_sec": sps,
                f"{name}/x_range_min": x_range_info[0],
                f"{name}/x_range_max": x_range_info[1],
                f"{name}/x_range_status": x_range_info[2],
            }
            print(f"[{name}] Step {step}/{max_steps} | Loss: {loss.item():.4f}")
            log_dict.update(_log_tensor_stats(f"{name}/x_t", x_t))
            log_dict.update(_log_tensor_stats(f"{name}/pred_x0", pred_x0))
            log_dict.update(_log_tensor_stats(f"{name}/noise_pred", noise_pred))
            print(f"[{name}] Step {step}/{max_steps} | Loss: {loss.item():.4f} | x_range={x_range_info}")
            if wandb.run is not None:
                wandb.log(log_dict,
                          step=step)  # Align step across models? usually better to use global step or just let wandb handle it.

        # -- Periodic Validation Sampling --
        if step % sample_every == 0 and step > 0:
            print(f"[{name}] Generating validation samples (Online vs EMA) at step {step}...")

            sample_targets = [("online", model)]
            if ema is not None:
                sample_targets.append(("ema", ema.shadow))
            for m_type, m_obj in sample_targets:
                m_obj.eval()
                with torch.no_grad():
                    val_samples = ddim_sample_single(m_obj, schedule, shape=(16, 1, 48, 48), device=device, steps=20)

                img_path = os.path.join(FLAGS.workdir, "samples", f"{name}_{m_type}_step_{step}.png")
                save_images_grid(val_samples, f"Val {name} {m_type.upper()}", img_path, nrow=4, log_wandb=True,
                                 step=step)

            model.train()

    return model, ema

def _combine_eps(eps4, eps7, w4, w7, normalize_eps=True, renormalize_sum=True,
                 per_timestep_norms=None, t=None):
    if normalize_eps:
        eps4 = eps4 / (eps4.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        eps7 = eps7 / (eps7.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
    if per_timestep_norms is not None and t is not None:
        norm4, norm7 = per_timestep_norms.get(int(t), (None, None))
        if norm4 is not None and norm7 is not None:
            target = 0.5 * (norm4 + norm7)
            eps4 = eps4 * (target / (norm4 + 1e-6))
            eps7 = eps7 * (target / (norm7 + 1e-6))
    eps = w4 * eps4 + w7 * eps7
    if renormalize_sum:
        eps = eps / (eps.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return eps, eps4, eps7


@torch.no_grad()
def compute_eps_norm_stats(model4, model7, schedule, steps, batch, device):
    model4.eval()
    model7.eval()
    shape = (batch, 1, 48, 48)
    x = torch.randn(shape, device=device)
    t_seq = torch.linspace(schedule.cfg.T - 1, 0, steps, device=device).long()
    norms = {}
    for t in t_seq:
        tt = torch.full((batch,), t.item(), device=device, dtype=torch.long)
        eps4 = model4(x, tt)
        eps7 = model7(x, tt)
        norm4 = float(eps4.flatten(1).norm(dim=1).mean().item())
        norm7 = float(eps7.flatten(1).norm(dim=1).mean().item())
        norms[int(t.item())] = (norm4, norm7)
    return norms

def describe_schedule(schedule: DiffusionSchedule):
    stats = {
        "beta_start": float(schedule.cfg.beta_start),
        "beta_end": float(schedule.cfg.beta_end),
        "T": int(schedule.cfg.T),
        "alpha_bar_0": float(schedule.alpha_bar[0].item()),
        "alpha_bar_Tm1": float(schedule.alpha_bar[-1].item()),
        "sqrt_one_minus_alpha_bar_max": float(schedule.sqrt_one_minus_alpha_bar.max().item()),
    }
    monotone = torch.all(schedule.alpha_bar[:-1] >= schedule.alpha_bar[1:]).item()
    stats["alpha_bar_monotone_decreasing"] = bool(monotone)
    return stats

def build_expert_dataset(cfg, mnist_train, digit, regime):
    if regime == "A":
        return PadTo48(filter_digit_subset(mnist_train, digit))
    if regime == "B":
        return TwoDigitMNISTCanvasClean(
            mnist_train,
            mode=f"exists{digit}",
            digit_size_range=cfg.digit_size_range,
            min_margin=cfg.min_margin,
        )
    if regime == "C":
        forbid_digit = digit if digit in (4, 7) else None
        target_overlap_digit = None
        if digit == 4:
            target_overlap_digit = 7
        elif digit == 7:
            target_overlap_digit = 4
        return TwoDigitMNISTCanvasCleanPlus(
            mnist_train,
            mode=f"exists{digit}",
            digit_size_range=cfg.digit_size_range,
            min_margin=cfg.min_margin,
            p_extra=cfg.get("p_extra", 0.3),
            forbid_digit=forbid_digit,
            target_overlap_digit=target_overlap_digit,
            target_overlap_prob=cfg.get("target_overlap_prob", 0.0),
        )
    raise ValueError(f"Unknown regime {regime}")

def build_constraint_dataset(cfg, mnist_train):
    return TwoDigitMNISTCanvasCleanPlus(
        mnist_train,
        mode=cfg.get("constraint_mode", "mixed"),
        digit_size_range=cfg.digit_size_range,
        min_margin=cfg.min_margin,
        p_extra=cfg.get("constraint_p_extra", cfg.get("p_extra", 0.3)),
        target_overlap_prob=cfg.get("target_overlap_prob", 0.0),
        return_counts=True,
    )


def load_expert_checkpoint(model, workdir, regime, digit):
    ckpt = os.path.join(workdir, f"regime_{regime}_expert{digit}.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        return ckpt
    return None

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
    schedule_cfg = DiffusionConfig(
        T=cfg.get("diffusion_T", 500),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
        prediction_type=cfg.get("prediction_type", "eps"),
    )
    if schedule_cfg.prediction_type != "eps":
        raise ValueError(f"Unsupported prediction_type {schedule_cfg.prediction_type}; PoE expects eps models.")
    schedule = DiffusionSchedule(schedule_cfg, device=device)
    schedule_stats = describe_schedule(schedule)
    print(f"Diffusion schedule: {schedule_stats}")
    if wandb.run is not None:
        wandb.log({f"schedule/{k}": v for k, v in schedule_stats.items()}, step=0)

    if cfg.regime == "C":
        cfg.target_overlap_digit = cfg.get("target_overlap_digit", None)
    ds4 = build_expert_dataset(cfg, mnist_train, 4, cfg.regime)
    ds7 = build_expert_dataset(cfg, mnist_train, 7, cfg.regime)

        # Optional toy single-digit override for quick sanity checks
    toy_digit = getattr(cfg, "toy_digit", None)
    if toy_digit is not None:
        toy_ds = PadTo48(filter_digit_subset(mnist_train, toy_digit))
        ds4 = toy_ds
        ds7 = toy_ds
    dl4 = DataLoader(ds4, batch_size=cfg.batch_size if 'batch_size' in cfg else 128, shuffle=True, num_workers=2,
                     pin_memory=True)
    dl7 = DataLoader(ds7, batch_size=cfg.batch_size if 'batch_size' in cfg else 128, shuffle=True, num_workers=2,
                     pin_memory=True)
    constraint_model = None
    constraint_use = cfg.get("use_count_constraint", False)
    if constraint_use:
        constraint_ds = build_constraint_dataset(cfg, mnist_train)
        constraint_dl = DataLoader(
            constraint_ds,
            batch_size=cfg.get("constraint_batch_size", cfg.batch_size if 'batch_size' in cfg else 128),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    anti_expert_digits = [int(d) for d in cfg.get("anti_expert_digits", [])]
    anti_dataloaders = {}
    if anti_expert_digits:
        for digit in anti_expert_digits:
            ds = build_expert_dataset(cfg, mnist_train, digit, cfg.regime)
            anti_dataloaders[digit] = DataLoader(
                ds,
                batch_size=cfg.batch_size if 'batch_size' in cfg else 128,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
    try:
        batch_dbg4, _ = next(iter(dl4))
        save_debug_batch(batch_dbg4, FLAGS.workdir, tag="train_batch_expert4", step=0, log_wandb=FLAGS.use_wandb)
        support4 = summarize_digit_support(ds4, draws=256)
        print(f"Digit support expert4 dataset (synthetic planner): {support4}")
        if wandb.run is not None:
            wandb.log({f"dataset/expert4_support_{k}": v for k, v in support4.items()}, step=0)
    except StopIteration:
        pass
    try:
        batch_dbg7, _ = next(iter(dl7))
        save_debug_batch(batch_dbg7, FLAGS.workdir, tag="train_batch_expert7", step=0, log_wandb=FLAGS.use_wandb)
        support7 = summarize_digit_support(ds7, draws=256)
        print(f"Digit support expert7 dataset (synthetic planner): {support7}")
        if wandb.run is not None:
            wandb.log({f"dataset/expert7_support_{k}": v for k, v in support7.items()}, step=0)
    except StopIteration:
        pass
    # -- Model Setup --
    if cfg.model_arch == "CenterBiasedExpert":
        model4 = CenterBiasedExpert(base=cfg.base_channels).to(device)
        model7 = CenterBiasedExpert(base=cfg.base_channels).to(device)
        anti_models = {digit: CenterBiasedExpert(base=cfg.base_channels).to(device) for digit in anti_expert_digits}
    else:
        model4 = FullyConvExpertBigger(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)
        model7 = FullyConvExpertBigger(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)
        anti_models = {
            digit: FullyConvExpertBigger(base=cfg.base_channels, n_blocks=cfg.get('n_blocks', 6)).to(device)
            for digit in anti_expert_digits
        }
        anti_emas = {}
    if constraint_use:
        constraint_model = CountConstraintNet().to(device)

    if FLAGS.mode == "train":
        print(f"Training Regime {cfg.regime} (Expert 4)...")
        m4, ema4 = train_expert_wandb(model4, dl4, schedule, device, max_steps=cfg.train_steps, name="expert4",
                                      sample_every=FLAGS.sample_every)
        torch.save(ema4.shadow.state_dict(), os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth"))

        print(f"Training Regime {cfg.regime} (Expert 7)...")
        m7, ema7 = train_expert_wandb(model7, dl7, schedule, device, max_steps=cfg.train_steps, name="expert7",
                                      sample_every=FLAGS.sample_every)
        torch.save(ema7.shadow.state_dict(), os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth"))

        if anti_models:
            for digit, model in anti_models.items():
                print(f"Training Regime {cfg.regime} (Anti-Expert {digit})...")
                dl = anti_dataloaders[digit]
                m, ema = train_expert_wandb(
                    model,
                    dl,
                    schedule,
                    device,
                    max_steps=cfg.train_steps,
                    name=f"expert{digit}",
                    sample_every=FLAGS.sample_every,
                )
                anti_emas[digit] = ema
                torch.save(ema.shadow.state_dict(),
                           os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert{digit}.pth"))

        if FLAGS.use_wandb:
            wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth"))
            wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth"))
            for digit in anti_expert_digits:
                wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert{digit}.pth"))
        if constraint_use:
            print("Training count constraint model...")
            constraint_model = train_count_constraint(
                constraint_model,
                constraint_dl,
                device,
                epochs=cfg.get("constraint_epochs", 3),
                lr=cfg.get("constraint_lr", 1e-3),
            )
            torch.save(
                constraint_model.state_dict(),
                os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_count_constraint.pth"),
            )
            if FLAGS.use_wandb:
                wandb.save(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_count_constraint.pth"))
    if FLAGS.mode == "eval" or FLAGS.mode == "train":
        if FLAGS.mode == "eval":
            print("Loading checkpoints...")
            model4.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert4.pth")))
            model7.load_state_dict(torch.load(os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_expert7.pth")))
            missing_digits = []
            for digit, model in anti_models.items():
                ckpt = load_expert_checkpoint(model, FLAGS.workdir, cfg.regime, digit)
                if ckpt:
                    print(f"Loaded anti-expert checkpoint for digit {digit}: {ckpt}")
                else:
                    print(f"Anti-expert checkpoint missing for digit {digit}; skipping.")
                    missing_digits.append(digit)
            for digit in missing_digits:
                anti_models.pop(digit, None)
            if constraint_use:
                constraint_ckpt = os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_count_constraint.pth")
                if os.path.exists(constraint_ckpt):
                    constraint_model.load_state_dict(torch.load(constraint_ckpt))
                else:
                    print("Count constraint checkpoint missing; disabling constraint guidance.")
                    constraint_use = False
        else:
            model4 = ema4.shadow if ema4 is not None else m4
            model7 = ema7.shadow if ema7 is not None else m7
            anti_models = {digit: ema.shadow for digit, ema in anti_emas.items()}
            if anti_models:
                print(f"Using EMA anti-experts: {list(anti_models.keys())}")
            if constraint_use and constraint_model is not None:
                constraint_model.eval()

        print("Training Evaluation Classifier...")
        clf = train_mnist_classifier(mnist_train, device)
        schedule_stats = describe_schedule(schedule)
        print("Schedule sanity:", schedule_stats, "| parameterization=eps")
        if wandb.run is not None:
            wandb.log({f"schedule/{k}": v for k, v in schedule_stats.items()})
            wandb.log({"schedule/parameterization": "eps"})

        print("Sampling PoE (Joint)...")
        poe_variants = [
            {"normalize_eps": True, "tag": "norm_eps"},
            {"normalize_eps": False, "tag": "raw_eps"},
        ]
        weight_sweep = cfg.get("weight_sweep", [(1.0, 1.0)])
        seeds = cfg.get("eval_seeds", [0])
        anti_weight = cfg.get("anti_weight", 1.0)

        for variant in poe_variants:
            for w4, w7 in weight_sweep:
                for seed in seeds:
                    x_poe, eps_stats = ddim_sample_poe_debug(
                        model4, model7, schedule,
                        shape=(64, 1, 48, 48),
                        steps=cfg.get("poe_steps", 100),
                        device=device,
                        normalize_eps=variant["normalize_eps"],
                        match_eps_norms=cfg.get("match_eps_norms", False),
                        anti_experts=anti_models if anti_models else None,
                        anti_weight=anti_weight,
                        seed=seed,
                        return_stats=True,
                        w4=w4,
                        w7=w7,
                        constraint_model=constraint_model if constraint_use else None,
                        constraint_weight=cfg.get("constraint_weight", 0.0),
                        target_counts=cfg.get("constraint_target_counts", (1.0, 1.0, 0.0)),
                    )

                    variant_tag = f"{variant['tag']}_w4{w4}_w7{w7}_s{seed}"
                    img_path = os.path.join(FLAGS.workdir, f"regime_{cfg.regime}_poe_final_{variant_tag}.png")
                    save_images_grid(x_poe, f"Regime {cfg.regime} Final PoE ({variant_tag})", img_path,
                                     log_wandb=FLAGS.use_wandb, step=cfg.train_steps)

                    metrics = eval_only_digits(x_poe, clf, device)
                    print(f"Regime {cfg.regime} Results ({variant_tag}):", metrics)

                    if FLAGS.use_wandb:
                        wandb.log({f"eval/{variant_tag}/{k}": v for k, v in metrics.items()})
                        if eps_stats:
                            wandb.log({f"eval/{variant_tag}/eps_stats": eps_stats})

if __name__ == "__main__":
    app.run(main)
