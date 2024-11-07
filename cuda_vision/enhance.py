import matplotlib.pyplot as plt
import torch
from __compile import load_src

_enhance = load_src("enhance")


def minmax_scale(img: torch.Tensor) -> torch.Tensor:
    return _enhance.minmax_scale(img)


def uniform_equalize(img: torch.Tensor, k) -> torch.Tensor:
    return _enhance.uniform_equalize(img, k)



if __name__ == "__main__":
    from utils.imageIO import *

    dark = load_raw('../imgs/rose_dark.raw', 256, 256, 1)
    mid = load_raw('../imgs/rose_mid.raw', 256, 256, 1)
    bright = load_raw('..//imgs/rose_bright.raw', 256, 256, 1)

    eq_dark64, eq_mid64, eq_bright64 = uniform_equalize(dark, 64), uniform_equalize(mid, 64), uniform_equalize(bright,
                                                                                                               64)
    compare_imgs([eq_dark64, eq_mid64, eq_bright64])
    plt.show()
