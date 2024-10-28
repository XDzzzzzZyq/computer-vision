import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

print(">>> Compiling CUDA Operators")

convert = load(
    "convert",
    sources=[
        os.path.join(module_path, "convert.cpp"),
        os.path.join(module_path, "convert_kernel.cu"),
    ],
)

def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return convert.to_grayscale(img)

if __name__ == "__main__":
    from utils.imgeIO import show_img, load_raw
    img = load_raw('../imgs/building_color.raw', 256, 256, 3)
    gray = to_grayscale(img)
    show_img(gray)
