import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "B"
    config.model_arch = "CenterBiasedExpert"
    config.data_mode = "clean_exists"
    config.base_channels = 64
    config.n_blocks = 6
    # Reduced for sanity check
    config.train_steps = 100
    config.digit_size_range = (20, 22)
    config.min_margin = 14
    return config