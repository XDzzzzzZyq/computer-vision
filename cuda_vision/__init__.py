import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)


def load_src(name, verbose=False):
    print(f"--->>> Compiling {name}")
    return load(name,
                sources=[os.path.join(module_path, f"{name}.cpp"),
                         os.path.join(module_path, f"{name}_kernel.cu")],
                extra_include_paths=[os.path.join(module_path, "include")],
                verbose=verbose)


print(">>> Compiling CUDA Operators")

convert = load_src("convert")
combine = load_src("combine")
enhance = load_src("enhance")
filter = load_src("filter")

print(">>> Finished")


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


def gaussian_conv(img: torch.Tensor, std, size, pad) -> torch.Tensor:
    assert std > 0.0, "std should be positive"
    return filter.gaussian_conv(img, std, size, pad)


def median_filter(img: torch.Tensor, size, pad) -> torch.Tensor:
    return filter.median_filter(img, size, pad, False)


def pseudo_median_filter(img: torch.Tensor, size, pad) -> torch.Tensor:
    return filter.median_filter(img, size, pad, True)


def bilateral_filter(img: torch.Tensor, std_s, std_i, size, pad) -> torch.Tensor:
    return filter.bilateral_filter(img, std_s, std_i, size, pad)


if __name__ == "__main__":
    from utils.imgeIO import *

    ori = load_raw('../imgs/rose_color.raw', 256, 256, 3)
    noi = load_raw('../imgs/rose_color_noise.raw', 256, 256, 3)
    noi_r, noi_g, noi_b = noi[:, 0:1], noi[:, 1:2], noi[:, 2:3]

    if True:
        filtered0 = pseudo_median_filter(noi_g, 20, 20)
        #filtered1 = pseudo_median_filter(noi_g, 1, 1)
        #filtered2 = pseudo_median_filter(noi_g, 15, 15)
        #filtered3 = median_filter(noi_g, 1, 1)
        filtered4 = median_filter(noi_g, 10, 10)
    else:
        filtered0 = bilateral_filter(noi_g, 1, 1, 0, 0)
        filtered1 = bilateral_filter(noi_g, 5, 10, 10, 10)
        filtered2 = bilateral_filter(noi_g, 5, 50, 10, 10)

    #compare_imgs([filtered0, filtered1, filtered2])
    compare_imgs([filtered0, filtered4])
    plt.show()
