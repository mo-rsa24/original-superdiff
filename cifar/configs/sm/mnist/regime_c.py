import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "C"
    config.model_arch = "FullyConvExpertBigger"
    config.data_mode = "clean_plus"
    config.base_channels = 96
    config.n_blocks = 8
    config.dilation_cycle = (1, 1, 2, 2, 3, 3, 4, 4)
    config.attn_at = 4
    config.attn_heads = 4
    config.dropout = 0.05
    config.train_steps = 500000
    config.batch_size = 128
    config.sample_steps = 120
    # Regime C Constraints
    config.digit_size_range = (18, 22)
    config.min_margin = 4
    return config
