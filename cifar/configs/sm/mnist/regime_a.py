import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "A"
    config.model_arch = "FullyConvExpertBigger"
    config.data_mode = "centered"
    config.base_channels = 96
    config.n_blocks = 6
    config.train_steps = 50000
    config.digit_size_range = (18, 22) # Default
    config.min_margin = 4 # Default
    config.batch_size = 128
    return config