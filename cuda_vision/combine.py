import torch
from cuda_vision.__compile import load_src

_combine = load_src("combine")


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _combine.overlay(a, b, 0)


def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _combine.overlay(a, b, 1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _combine.overlay(a, b, 2)


def divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _combine.overlay(a, b, 3)


def length(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _combine.overlay(x, y, 4)


def land(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from cuda_vision import convert
    c = multiply(a, b)
    c = convert.binarize(c, 254.9)
    return c


def lor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from cuda_vision import convert
    c = add(a, b)
    c = convert.binarize(c, 254.9)
    return c


def clamp(img: torch.Tensor, low, high) -> torch.Tensor:
    return _combine.clamp(img, low, high)


def make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0)) -> torch.Tensor:
    return _combine.watermark(img, mark, offset[0], offset[1])
