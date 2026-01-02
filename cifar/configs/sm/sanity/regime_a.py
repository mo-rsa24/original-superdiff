import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "A"
    config.model_arch = "FullyConvExpertBigger"
    config.data_mode = "centered"
    config.base_channels = 96
    config.n_blocks = 6
    # Reduced for sanity check
    config.train_steps = 100
    config.digit_size_range = (18, 22)
    config.min_margin = 4
    config.batch_size = 128
    return config