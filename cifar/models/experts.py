import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.to_time = nn.Linear(tdim, c)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.to_time(t_emb).view(-1, h.size(1), 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h
class MultiDilatedBlock(nn.Module):
    """
    A lightweight multi-dilation bottleneck that expands the receptive field
    without resorting to pooling or positional encodings. Parallel dilated
    branches aggregate evidence across the 48x48 canvas while remaining fully
    convolutional and translation-equivariant.
    """

    def __init__(self, c, tdim, dilation_rates=(1, 2, 3)):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.to_time = nn.Linear(tdim, c)
        self.branches = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=r, dilation=r) for r in dilation_rates
        ])
        self.mixer = nn.Conv2d(c * len(dilation_rates), c, 1)

    def forward(self, x, t_emb):
        h = F.silu(self.norm(x))
        h = h + self.to_time(t_emb).view(-1, h.size(1), 1, 1)
        feats = [F.silu(branch(h)) for branch in self.branches]
        mixed = self.mixer(torch.cat(feats, dim=1))
        return x + mixed

class FullyConvExpertBigger(nn.Module):
    def __init__(self, in_ch=1, base=96, tdim=128, n_blocks=6,
                 dilation_rates=(1, 2, 3), post_dilated_blocks=2):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.dilated = MultiDilatedBlock(base, tdim, dilation_rates=dilation_rates)
        self.post_blocks = nn.ModuleList([ResBlock(base, tdim) for _ in range(post_dilated_blocks)])
        self.mid = nn.Conv2d(base, base, 3, padding=1)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))
        h = F.silu(self.in_conv(x))
        for b in self.blocks:
            h = b(h, t_emb)
        h = self.dilated(h, t_emb)
        for b in self.post_blocks:
            h = b(h, t_emb)
        h = F.silu(self.mid(h))
        return self.out(h)

class CenterBiasedExpert(nn.Module):
    """
    Regime B Arch: Compresses spatial information to a global vector,
    forcing the model to learn coordinate-dependent (centered) features.
    """
    def __init__(self, in_ch=1, base=64, tdim=128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1), nn.SiLU(),      # 24x24
            nn.Conv2d(base, base*2, 3, 2, 1), nn.SiLU(),     # 12x12
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.SiLU(),   # 6x6
            nn.Flatten()                                     # Global aggregation
        )

        flat_dim = base * 4 * 6 * 6
        self.mid = nn.Sequential(
            nn.Linear(flat_dim + tdim, 1024), nn.SiLU(),
            nn.Linear(1024, 1024), nn.SiLU(),
            nn.Linear(1024, flat_dim), nn.SiLU()
        )

        self.up = nn.Sequential(
            nn.Unflatten(1, (base*4, 6, 6)),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(base, in_ch, 4, 2, 1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))
        h = self.down(x)
        h = torch.cat([h, t_emb], dim=1)
        h = self.mid(h)
        return self.up(h)

class SmallMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)