import matplotlib.pyplot as plt
import torch
from typing import overload
from cuda_vision.__compile import load_src

_stats = load_src("stats")


def get_sat(imgs: torch.Tensor, order=4) -> torch.Tensor:
    return _stats.to_sat(imgs, order)


if __name__ == "__main__":
    from utils.imageIO import *

    tex = load_raw('../imgs/comb1.raw', 256, 256, 1)
    a = _stats.to_sat(tex, 4)
    compare_imgs([a[:,0:1], a[:,1:2]], range=None)
    plt.show()
