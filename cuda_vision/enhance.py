import torch
from __compile import load_src

enhance = load_src("enhance")


def minmax_scale(img: torch.Tensor) -> torch.Tensor:
    return enhance.minmax_scale(img)


def uniform_equalize(img: torch.Tensor, k) -> torch.Tensor:
    return enhance.uniform_equalize(img, k)
