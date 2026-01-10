import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0

    # experiment settings
    config.experiment = "attribute_composition"
    config.target_digit = 7
    config.target_color = "blue"
    config.color_palette = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    config.color_jitter = 0.1

    # data/model
    config.image_size = 28
    config.num_channels = 3
    config.model_arch = "FullyConvExpertBigger"
    config.base_channels = 96
    config.n_blocks = 6

    # diffusion schedule
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.T = 500
    config.diffusion.beta_start = 1e-4
    config.diffusion.beta_end = 0.02

    # training
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 128
    config.train.steps = 50000
    config.train.lr = 2e-4
    config.train.log_every = 100
    config.train.sample_every = 5000

    # sampling/eval
    config.sampling = ml_collections.ConfigDict()
    config.sampling.steps = 100
    config.sampling.eta = 0.0
    config.sampling.normalize_eps = True
    config.sampling.renormalize_sum = True
    config.sampling.num_samples = 64

    # composition regimes
    config.regimes = ["shape_only", "color_only", "shape_and_color"]
    config.enable_superdiff_and = True

    return config