import matplotlib.pyplot as plt
import torch
from typing import overload
from cuda_vision.__compile import load_src

_stats = load_src("stats")


def get_sat(imgs: torch.Tensor, order=4) -> torch.Tensor:
    return _stats.to_sat(imgs, order)


def get_momentum_features(imgs: torch.Tensor, order=3, window=4, sat=None) -> torch.Tensor:
    if sat is None:
        sat = get_sat(imgs, order)
    assert window > 1
    return _stats.get_momentum(sat, window)


def get_minmaxmedian_features(imgs: torch.Tensor, window=4) -> torch.Tensor:
    assert window > 1
    return _stats.get_minmaxmedian(imgs, window)


def get_laws_features(imgs: torch.Tensor, window=4) -> torch.Tensor:
    from cuda_vision.__kernels import get_laws_kernel
    from cuda_vision.filters import custom_multi_conv
    assert window > 1
    kernels = get_laws_kernel().to(imgs.device)
    B, _, H, W = imgs.shape
    features = custom_multi_conv(imgs, kernels, 1) # B, 9, H, W
    features = features.reshape(B*9, 1, H, W)
    features = get_momentum_features(features, 2, window) # B*9, 2, H//window, W//window
    return features.reshape(B, 9, 2, H//window, W//window)[:, :, 1]


def get_full_features(imgs: torch.Tensor, window=4, normalize=True, sat=None) -> torch.Tensor:
    momentum_features = get_momentum_features(imgs, window=window, order=4, sat=sat)
    minmaxmedian_features = get_minmaxmedian_features(imgs, window=window)
    laws_features = get_laws_features(imgs, window=window)
    features = torch.cat([momentum_features, minmaxmedian_features, laws_features], dim=1)
    assert features.ndim == 4
    assert features.shape[1] == 4 + 3 + 9
    if normalize:
        from cuda_vision.enhance import minmax_scale
        B, F, H, W = features.shape
        features = features.reshape(B*F, 1, H, W)
        features = minmax_scale(features)
        features = features.reshape(B, F, H, W)
    return features


if __name__ == "__main__":
    from utils.imageIO import *

    tex = load_raw('../imgs/comb1.raw', 256, 256, 1)
    a = get_full_features(tex, 8, normalize=True)
    list = [a[:, t:t+1] for t in range(a.shape[1])]
    compare_imgs_grid(list, shape=(4, 4))
    plt.show()
