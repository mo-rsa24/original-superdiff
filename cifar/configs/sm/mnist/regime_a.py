from .regime_common import build_regime_config

def get_config():
    return build_regime_config(
        train_split='train[:50%]',
        eval_split='test',
        nf=96,
        n_iters=50000,
        regime_label='A',
    )