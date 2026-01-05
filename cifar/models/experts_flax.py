import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from . import layers
from . import utils


def _time_embedding(t, dim):
  temb = layers.get_timestep_embedding(t.reshape((-1,)), dim)
  temb = nn.Dense(dim)(temb)
  temb = nn.silu(temb)
  temb = nn.Dense(dim)(temb)
  return temb


class _ResBlock(nn.Module):
  features: int
  time_dim: int
  groups: int = 8

  @nn.compact
  def __call__(self, x, temb, train: bool):
    h = nn.GroupNorm(num_groups=self.groups)(x)
    h = nn.silu(h)
    h = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(h)
    h = h + nn.Dense(self.features)(nn.silu(temb))[:, None, None, :]
    h = nn.GroupNorm(num_groups=self.groups)(h)
    h = nn.silu(h)
    h = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(h)
    return x + h


class _MultiDilatedBlock(nn.Module):
  features: int
  time_dim: int
  dilation_rates: Sequence[int] = (1, 2, 3)
  groups: int = 8

  @nn.compact
  def __call__(self, x, temb, train: bool):
    h = nn.GroupNorm(num_groups=self.groups)(x)
    h = nn.silu(h)
    h = h + nn.Dense(self.features)(nn.silu(temb))[:, None, None, :]
    feats = []
    for rate in self.dilation_rates:
      branch = nn.Conv(
          self.features,
          kernel_size=(3, 3),
          padding="SAME",
          dilation=(rate, rate),
      )(h)
      feats.append(nn.silu(branch))
    h = nn.Conv(
        self.features,
        kernel_size=(1, 1),
        padding="SAME",
    )(jnp.concatenate(feats, axis=-1))
    return x + h


@utils.register_model(name="fullyconv-expert-bigger")
class FullyConvExpertBigger(nn.Module):
  """Fully convolutional expert used in regimes A and C."""

  config: any

  @nn.compact
  def __call__(self, t, x, y, train: bool):
    del y  # Unused label conditioning for MNIST expert regimes.
    config = self.config
    base = config.model.nf
    time_dim = 128
    n_blocks = getattr(config.model, "num_res_blocks", 6)
    post_blocks = getattr(config.model, "post_res_blocks", 2)
    dilation_rates = getattr(config.model, "dilation_rates", (1, 2, 3))

    temb = _time_embedding(t, time_dim)
    h = nn.Conv(base, kernel_size=(3, 3), padding="SAME")(x)
    h = nn.silu(h)

    res_block = functools.partial(_ResBlock, features=base, time_dim=time_dim)
    for _ in range(n_blocks):
      h = res_block()(h, temb, train)

    h = _MultiDilatedBlock(
        features=base,
        time_dim=time_dim,
        dilation_rates=dilation_rates,
    )(h, temb, train)

    for _ in range(post_blocks):
      h = res_block()(h, temb, train)

    h = nn.Conv(base, kernel_size=(3, 3), padding="SAME")(nn.silu(h))
    h = nn.Conv(
        config.data.num_channels,
        kernel_size=(3, 3),
        padding="SAME",
        kernel_init=layers.default_init(),
    )(h)
    return h


@utils.register_model(name="center-biased-expert")
class CenterBiasedExpert(nn.Module):
  """Center-biased expert for regime B."""

  config: any

  @nn.compact
  def __call__(self, t, x, y, train: bool):
    del y  # Unused label conditioning for MNIST expert regimes.
    config = self.config
    base = config.model.nf
    time_dim = 128

    temb = _time_embedding(t, time_dim)

    h = nn.Conv(base, (3, 3), strides=(2, 2), padding="SAME")(x)
    h = nn.silu(h)
    h = nn.Conv(base * 2, (3, 3), strides=(2, 2), padding="SAME")(h)
    h = nn.silu(h)
    h = nn.Conv(base * 4, (3, 3), strides=(2, 2), padding="SAME")(h)
    h = nn.silu(h)

    flat = h.reshape((h.shape[0], -1))
    h = jnp.concatenate([flat, temb], axis=-1)
    h = nn.Dense(1024)(h)
    h = nn.silu(h)
    h = nn.Dense(1024)(h)
    h = nn.silu(h)
    flat_dim = base * 4 * 6 * 6
    h = nn.Dense(flat_dim)(h)
    h = nn.silu(h)
    h = h.reshape((h.shape[0], 6, 6, base * 4))

    h = nn.ConvTranspose(base * 2, (4, 4), strides=(2, 2), padding="SAME")(h)
    h = nn.silu(h)
    h = nn.ConvTranspose(base, (4, 4), strides=(2, 2), padding="SAME")(h)
    h = nn.silu(h)
    h = nn.ConvTranspose(
        config.data.num_channels, (4, 4), strides=(2, 2), padding="SAME"
    )(h)
    return h