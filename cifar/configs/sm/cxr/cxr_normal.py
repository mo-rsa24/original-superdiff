import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'ChestXRay'
    data.root_dir = '../datasets/cleaned/TB/train'  # Ensure this path is correct on your cluster
    data.target_class = 'NORMAL'  # <--- Trains only on NORMAL
    data.ndims = 3
    data.image_size = 64  # X-Rays need higher res, but start small (32 or 64) for speed
    data.num_channels = 1
    data.uniform_dequantization = True
    data.random_flip = True  # Valid for X-Rays
    data.task = 'generate'
    data.dynamics = 'generation'  # or vpsde
    data.t_0, data.t_1 = 0.0, 1.0

    # model (Standard configuration)
    config.model = model = ml_collections.ConfigDict()
    model.name = 'anet'  # or 'score-net' depending on preference
    model.loss = 'sam'
    model.sigma = 1e-1
    model.anneal_sigma = False
    model.ema_rate = 0.99
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 64  # Increased filters for X-Ray complexity
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.dropout = 0.1

    # training
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 64  # Adjust based on GPU memory
    train.n_jitted_steps = 1
    train.n_iters = 100_000
    train.save_every = 5_000
    train.eval_every = 5_000
    train.log_every = 50
    train.lr = 2e-4
    train.beta1 = 0.9
    train.eps = 1e-8
    train.warmup = 1_000
    train.grad_clip = 1.

    # evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 64
    eval.artifact_size = 32
    eval.num_samples = 500
    eval.use_ema = False

    return config