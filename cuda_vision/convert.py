import torch
from __compile import load_src

convert = load_src("convert")


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return convert.to_grayscale(img)


def invert(img: torch.Tensor) -> torch.Tensor:
    return convert.invert(img)