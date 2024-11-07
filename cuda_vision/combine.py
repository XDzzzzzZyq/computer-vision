import torch
from __compile import load_src

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


def make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0)) -> torch.Tensor:
    return _combine.watermark(img, mark, offset[0], offset[1])
