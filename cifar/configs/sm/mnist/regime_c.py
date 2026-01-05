from .regime_common import build_regime_config

def get_config():
    return build_regime_config(
        train_split='train>5',
        eval_split='test>5',
        nf=96,
        n_iters=500_000,
        regime_label='C',
        data_mode='clean_plus',
        model_name='fullyconv-expert-bigger',
    )