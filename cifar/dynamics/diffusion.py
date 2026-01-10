import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class DiffusionConfig:
    T: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02

class DiffusionSchedule:
    def __init__(self, cfg: DiffusionConfig, device):
        self.cfg = cfg
        T = cfg.T
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alpha_bar[t].view(-1,1,1,1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1)
        return s1*x0 + s2*noise, noise

@torch.no_grad()
def ddim_sample(model, schedule: DiffusionSchedule, shape, steps=100, eta=0.0, seed=0, device='cuda'):
    torch.manual_seed(seed)
    model.eval()
    x = torch.randn(shape, device=device)
    T = schedule.cfg.T
    t_seq = torch.linspace(T-1, 0, steps, device=device).long()

    for i in range(len(t_seq)):
        t = t_seq[i].item()
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        abar_t = schedule.alpha_bar[t]
        x0 = (x - torch.sqrt(1.0 - abar_t) * eps) / (torch.sqrt(abar_t) + 1e-8)

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
    return x

@torch.no_grad()
def ddim_sample_poe(model4, model7, schedule: DiffusionSchedule, shape, steps=100, eta=0.0,
                    w4=1.0, w7=1.0, normalize_eps=True, renormalize_sum=True,
                    seed=0, device='cuda'):
    torch.manual_seed(seed)
    model4.eval(); model7.eval()
    x = torch.randn(shape, device=device)
    T = schedule.cfg.T
    t_seq = torch.linspace(T-1, 0, steps, device=device).long()

    for i in range(len(t_seq)):
        t = t_seq[i].item()
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)

        eps4 = model4(x, tt)
        eps7 = model7(x, tt)
        if normalize_eps:
            eps4 = eps4 / (eps4.std(dim=(1,2,3), keepdim=True) + 1e-6)
            eps7 = eps7 / (eps7.std(dim=(1,2,3), keepdim=True) + 1e-6)

        eps = w4*eps4 + w7*eps7
        if renormalize_sum:
            eps = eps / (eps.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        abar_t = schedule.alpha_bar[t]
        x0 = (x - torch.sqrt(1.0 - abar_t) * eps) / (torch.sqrt(abar_t) + 1e-8)

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
    return x