import ml_collections


# SAME AS ABOVE, but different target class
def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'
    data.train_split = 'train'
    data.target_class = 9  # <--- Model B trains on Class 9 (Trucks)

    # ... Copy rest from vpsdeA_fast.py ...
    data.ndims = 3
    data.image_size = 32
    data.num_channels = 3
    data.num_classes = 10
    data.uniform_dequantization = True
    data.random_flip = True
    data.task = 'generate'
    data.dynamics = 'vpsde'
    data.t_0, data.t_1 = 0.0, 1.0

    config.model = model = ml_collections.ConfigDict()
    model.name = 'score-net'
    model.conditioned = True
    model.loss = 'dsm'
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16, 8)
    model.resamp_with_conv = True
    model.dropout = 0.1

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 128
    train.n_jitted_steps = 5
    train.n_iters = 100000  # Good medium-length run
    train.save_every = 10000  # Saves 10 checkpoints
    train.eval_every = 10000  # Evaluates every checkpoint (aligned with save)
    train.log_every = 100  # Vital: frequent enough to see loss curve
    train.lr = 2e-4
    train.beta1 = 0.9
    train.eps = 1e-8
    train.warmup = 5000
    train.grad_clip = 1.

    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 500
    eval.artifact_size = 64
    eval.num_samples = 10000
    eval.use_ema = True
    eval.estimate_bpd = True

    return config