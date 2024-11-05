import torch
from __compile import load_src

combine = load_src("combine")


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return combine.overlay(a, b, 0)


def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return combine.overlay(a, b, 1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return combine.overlay(a, b, 2)


def divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return combine.overlay(a, b, 3)


def make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0)) -> torch.Tensor:
    return combine.watermark(img, mark, offset[0], offset[1])
