from typing import Optional

import ml_collections


def build_regime_config(
    train_split: str,
    eval_split: str,
    *,
    seed: int = 1,
    nf: int = 96,
    n_iters: int = 50_000,
    ema_rate: float = 0.9999,
    regime_label: Optional[str] = None,
    data_mode: Optional[str] = None,
    model_name: str = "score-net",
):
  """Constructs a SuperDiff-style config for MNIST regimes."""
  config = ml_collections.ConfigDict()
  config.seed = seed
  if regime_label is not None:
    config.regime = regime_label

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.train_split = train_split
  data.eval_split = eval_split
  if data_mode is not None:
    data.mode = data_mode
  data.ndims = 3
  data.image_size = 32
  data.num_channels = 1
  data.num_classes = 10
  data.uniform_dequantization = True
  data.random_flip = False
  data.task = 'generate'
  data.dynamics = 'vpsde'
  data.t_0, data.t_1 = 0.0, 1.0

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = model_name
  model.conditioned = False
  model.loss = 'dsm'
  model.ema_rate = ema_rate
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = nf
  model.ch_mult = (1, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.dropout = 0.1

  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 128
  train.n_jitted_steps = 1
  train.n_iters = n_iters
  train.save_every = 5_000
  train.eval_every = 5_000
  train.log_every = 50
  train.lr = 2e-4
  train.beta1 = 0.9
  train.eps = 1e-8
  train.warmup = 5_000
  train.grad_clip = 1.

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 128
  eval.artifact_size = 64
  eval.num_samples = 10_000
  eval.use_ema = True
  eval.estimate_bpd = False

  return config