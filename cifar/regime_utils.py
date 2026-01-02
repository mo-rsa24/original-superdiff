import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class EMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

def train_expert_steps(model, dl, schedule, device, max_steps=50000, lr=2e-4, grad_clip=1.0, ema_decay=0.9999, log_every=500):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)
    model.train()

    losses = []
    t0 = time.time()
    it = iter(dl)

    for step in range(1, max_steps + 1):
        try:
            x0, _ = next(it)
        except StopIteration:
            it = iter(dl)
            x0, _ = next(it)

        x0 = x0.to(device)
        B = x0.size(0)
        t = torch.randint(0, schedule.cfg.T, (B,), device=device)
        xt, eps = schedule.q_sample(x0, t)

        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        ema.update(model)
        losses.append(loss.item())

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{max_steps} | loss={np.mean(losses[-log_every:]):.4f} | time={dt:.1f}s")

    return model, ema, losses