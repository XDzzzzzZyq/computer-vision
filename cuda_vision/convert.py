import torch
from cuda_vision.__compile import load_src

_convert = load_src("convert")


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return _convert.to_grayscale(img)


def invert(img: torch.Tensor) -> torch.Tensor:
    return _convert.invert(img)


def binarize(img: torch, threshold) -> torch.Tensor:
    return _convert.binarize(img, threshold)
