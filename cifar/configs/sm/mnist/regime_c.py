import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "C"
    config.model_arch = "FullyConvExpertBigger"
    config.data_mode = "clean_plus"
    config.base_channels = 96
    config.n_blocks = 6
    config.train_steps = 50000
    config.batch_size = 128
    # Regime C Constraints
    config.digit_size_range = (18, 22)
    config.min_margin = 4
    return config