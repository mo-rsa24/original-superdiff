import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.regime = "C"
    config.model_arch = "FullyConvExpertBigger"
    config.data_mode = "clean_plus"
    config.base_channels = 128
    config.n_blocks = 8
    config.train_steps = 650000
    config.batch_size = 128
    config.use_ema = True
    config.ema_decay = 0.9995
    config.diffusion_T = 500
    config.beta_start = 1e-4
    config.beta_end = 0.02
    config.prediction_type = "eps"
    # Regime C Constraints
    config.digit_size_range = (18, 22)
    config.min_margin = 4
    config.p_extra = 0.4
    config.target_overlap_prob = 0.35
    config.poe_steps = 150
    config.match_eps_norms = True
    config.anti_expert_digits = [0, 1, 2, 3, 5, 6, 8, 9]
    config.anti_weight = 0.7
    config.weight_sweep = [(0.75, 1.25), (1.0, 1.0), (1.25, 0.75)]
    config.eval_seeds = [0, 1]
    return config
