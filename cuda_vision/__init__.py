import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)


def load_src(name):
    return  load(name,
                 sources=[
                     os.path.join(module_path, f"{name}.cpp"),
                     os.path.join(module_path, f"{name}_kernel.cu")
                 ],
                 extra_include_paths=[os.path.join(module_path, "include")])


print(">>> Compiling CUDA Operators")

convert = load_src("convert")
combine = load_src("combine")
enhance = load_src("enhance")
filter = load_src("filter")


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return convert.to_grayscale(img)


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


def invert(img: torch.Tensor) -> torch.Tensor:
    return convert.invert(img)


def minmax_scale(img: torch.Tensor) -> torch.Tensor:
    return enhance.minmax_scale(img)


def uniform_equalize(img: torch.Tensor, k) -> torch.Tensor:
    return enhance.uniform_equalize(img, k)


def uniform_conv(img: torch.Tensor, size, pad) -> torch.Tensor:
    return filter.uniform_conv(img, size, pad)


if __name__ == "__main__":
    from utils.imgeIO import *

    gau = load_raw('../imgs/rose_gau.raw', 256, 256, 1)
    uni = load_raw('../imgs/rose_uni.raw', 256, 256, 1)
    ori = load_raw('../imgs/rose.raw', 256, 256, 1)

    conv = uniform_conv(ori, 3, 3)
    compare_imgs([ori, conv])
    print(conv)
    plt.show()



