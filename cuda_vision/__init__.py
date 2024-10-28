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


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return convert.to_grayscale(img)


def make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0)) -> torch.Tensor:
    return combine.watermark(img, mark, offset[0], offset[1])


def invert(img: torch.Tensor) -> torch.Tensor:
    return convert.invert(img)


if __name__ == "__main__":
    from utils.imgeIO import compare_imgs, show_img, load_raw
    import matplotlib.pyplot as plt

    img = load_raw('../imgs/building2_color.raw', 256, 256, 3)
    gray = to_grayscale(img)
    gray_gt = 0.299*img[:,0:1] + 0.587*img[:,1:2] + 0.114*img[:,2:3]
    compare_imgs([img, gray])
    print(torch.allclose(gray, gray_gt))

    mark = load_raw('../imgs/dku_logo_color.raw', 128, 128, 3)
    marked_img = make_watermark(img, mark)
    compare_imgs([img, marked_img])
    plt.show()

    inverted_img = invert(img)
    compare_imgs([img, inverted_img])
    plt.show()
