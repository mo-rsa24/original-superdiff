import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "B"
    config.model_arch = "CenterBiasedExpert"
    config.data_mode = "clean_exists"
    config.base_channels = 64 # CenterBiased uses 64 default
    config.n_blocks = 6 # Ignored by CenterBiased
    config.train_steps = 50000
    config.digit_size_range = (20, 22) # Force restricted range
    config.min_margin = 14 # Force center bias
    return config