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
    def __init__(self, c, tdim, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dilation = int(dilation)
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=self.dilation, dilation=self.dilation)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=self.dilation, dilation=self.dilation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.to_time = nn.Linear(tdim, c)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.to_time(t_emb).view(-1, h.size(1), 1, 1)
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

        @property
        def receptive_field_contribution(self):
            # Each 3x3 conv with dilation d increases RF by (k-1)*d = 2d.
            return 4 * self.dilation

class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention that preserves translation equivariance.
    Uses 1x1 convolutions for QKV projections and no positional encodings.
    """

    def __init__(self, c: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert c % num_heads == 0, "channels must be divisible by heads"
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1)
        self.out = nn.Conv2d(c, c, kernel_size=1)
        self.norm = nn.GroupNorm(8, c, eps=eps)

    def forward(self, x):
        b, c, h, w = x.shape
        h_w = h * w
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h_w).transpose(2, 3)  # [b, heads, hw, d]
        k = k.reshape(b, self.num_heads, self.head_dim, h_w)  # [b, heads, d, hw]
        v = v.reshape(b, self.num_heads, self.head_dim, h_w).transpose(2, 3)  # [b, heads, hw, d]

        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # [b, heads, hw, hw]
        out = torch.matmul(attn, v)  # [b, heads, hw, d]
        out = out.transpose(2, 3).reshape(b, c, h, w)
        return self.out(out)

class FullyConvExpertBigger(nn.Module):
    def __init__(
            self,
            in_ch=1,
            base=96,
            tdim=128,
            n_blocks=6,
            dilation_cycle=(1, 1, 2, 2, 3, 3),
            attn_at=None,
            attn_heads=4,
            dropout=0.0,
    ):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.block_dilations = [dilation_cycle[i % len(dilation_cycle)] for i in range(n_blocks)]
        self.blocks = nn.ModuleList([ResBlock(base, tdim, dilation=d, dropout=dropout) for d in self.block_dilations])
        self.attn_at = int(attn_at) if attn_at is not None else None
        self.attn = SpatialSelfAttention(base, num_heads=attn_heads) if self.attn_at is not None else None
        self.mid = nn.Conv2d(base, base, 3, padding=2, dilation=2)
        self.mid_norm = nn.GroupNorm(8, base)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))
        h = F.silu(self.in_conv(x))
        for idx, b in enumerate(self.blocks):
            h = b(h, t_emb)
            if self.attn is not None and idx == self.attn_at:
                h = h + self.attn(h)
        h = F.silu(self.mid_norm(self.mid(h)))
        h = F.silu(self.out_norm(h))
        return self.out(h)

    def estimate_receptive_field(self):
        """
        Rough receptive field estimate assuming stride=1 everywhere.
        """
        rf = 1  # single pixel
        rf += 2  # in_conv (3x3)
        for d in self.block_dilations:
            rf += 2 * d  # conv1
            rf += 2 * d  # conv2
        rf += 2 * self.mid.dilation[0]  # mid conv dilation increases field
        rf += 2  # final 3x3
        return rf

    def describe(self):
        return {
            "n_blocks": len(self.blocks),
            "block_dilations": list(self.block_dilations),
            "attn_at": self.attn_at,
            "attn_heads": self.attn.num_heads if self.attn is not None else 0,
            "estimated_receptive_field": self.estimate_receptive_field(),
        }

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