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


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return convert.to_grayscale(img)


def make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0)) -> torch.Tensor:
    return combine.watermark(img, mark, offset[0], offset[1])


def invert(img: torch.Tensor) -> torch.Tensor:
    return convert.invert(img)


def minmax_scale(img: torch.Tensor) -> torch.Tensor:
    return enhance.minmax_scale(img)


def uniform_equalize(img: torch.Tensor, k) -> torch.Tensor:
    return enhance.uniform_equalize(img, k)


if __name__ == "__main__":
    from utils.imgeIO import *

    img = load_raw('../imgs/rose_dark.raw', 256, 256, 1)
    scaled_img = minmax_scale(img)
    eqlzed_img8  = uniform_equalize(img, 8)
    eqlzed_img64 = uniform_equalize(img, 64)
    eqlzed_img200= uniform_equalize(img, 200)
    compare_imgs([img, scaled_img, eqlzed_img8, eqlzed_img64, eqlzed_img200])
    compare_hist([img, scaled_img, eqlzed_img8, eqlzed_img64, eqlzed_img200], bins=35)
    plt.show()
