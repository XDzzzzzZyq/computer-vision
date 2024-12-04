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
    B, _, W, H = imgs.shape
    features = custom_multi_conv(imgs, kernels, 1) # B, 9, W, H
    features = features.reshape(B*9, 1, W, H)
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
        B, F, W, H = features.shape
        features = features.reshape(B*F, 1, W, H)
        features = minmax_scale(features)
        features = features.reshape(B, F, W, H)
    return features


def get_metric_properties(imgs: torch.Tensor) -> torch.Tensor:
    assert imgs.ndim == 4
    assert imgs.shape[2] % 2 == imgs.shape[3] % 2 == 0
    from cuda_vision.__kernels import get_metric_patterns
    import math
    patterns = get_metric_patterns().to(imgs.device)
    qs = _stats.get_metric_properties(imgs, patterns).float()
    nsq = 1/math.sqrt(2)
    weights = torch.tensor([[.0, .25, .25, .25, .25, .5, .5, .5, .5, .875, .875, .875, .875, 1., .75, .75],
                            [.0, nsq, nsq, nsq, nsq, 1., 1., 1., 1., nsq, nsq, nsq, nsq, .0, 2*nsq, 2*nsq]])
    ap = torch.matmul(weights.to(qs.device)[None, :, :], qs[:, :, None]).squeeze(2)
    r = ap[:, 0:1] / ap[:, 1:2]
    return torch.cat([ap, r], dim=1)


def get_holes_num(imgs: torch.Tensor, max_iter=50) -> torch.Tensor:
    from cuda_vision.convert import invert
    from scipy.ndimage import label
    imgs = invert(imgs)
    num = [label(img.cpu())[1]-1 for img in imgs]
    return torch.tensor(num).float().unsqueeze(1).to(imgs.device)


if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.convert import invert
    from cuda_vision.transform import island_segment

    train = load_raw('../imgs/training.raw', 256, 256, 3)[:, 0:1].contiguous()
    train = invert(train)
    segments, ratio = island_segment(train, pad=2)
    segments = torch.cat(segments, dim=0)
    segments = get_holes_num(segments)
    print(segments)
