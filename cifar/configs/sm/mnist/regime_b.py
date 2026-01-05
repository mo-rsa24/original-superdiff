from .regime_common import build_regime_config

def get_config():
    return build_regime_config(
        train_split='train[50%:]',
        eval_split='test',
        nf=64,
        n_iters=50_000,
        regime_label='B',
        data_mode='clean_exists',
        model_name='center-biased-expert',
    )