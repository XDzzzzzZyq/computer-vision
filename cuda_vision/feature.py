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
    return _stats.get_momentum(sat, window)


def get_minmax_features(imgs: torch.Tensor, window=4) -> torch.Tensor:
    pass
    #return _stats.get_minmax(imgs, window)


def get_laws_features(imgs: torch.Tensor, window=4) -> torch.Tensor:
    pass


def get_full_features(imgs: torch.Tensor, window=4, normalize=True, sat=None) -> torch.Tensor:
    momentum_features = get_momentum_features(imgs, window=window, order=4, sat=sat)
    minmax_features = get_minmax_features(imgs, window=window)
    laws_features = get_laws_features(imgs, window=window)
    features = torch.cat([momentum_features, minmax_features, laws_features], dim=1)
    assert features.ndim == 4
    assert features.shape[1] == 4 + 2 + 9*2
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
    a = get_momentum_features(tex, 3, 8)
    compare_imgs([tex, a[:,0:1], a[:,1:2], a[:,2:3]], range=None)
    plt.show()
