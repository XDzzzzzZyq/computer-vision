import torch
from cuda_vision.__compile import load_src

_transform = load_src("transform")


class Transform(object):
    def __init__(self, offset=(0.0,0.0), angle=0.0, ratio=(1.0,1.0)):
        self.offset = offset
        self.angle = angle
        self.ratio = ratio


def translate(imgs: torch.Tensor, offset: tuple[float, float]) -> torch.Tensor:
    return _transform.simple_transform(imgs, offset[0], offset[1], 0)


def rotate(imgs: torch.Tensor, angle: float) -> torch.Tensor:
    return _transform.simple_transform(imgs, angle, 0.0, 1)


def scale(imgs: torch.Tensor, ratio: tuple[float, float]) -> torch.Tensor:
    if ratio[0] == 0.0 or ratio[1] == 0.0:
        return torch.zeros_like(imgs)
    return _transform.simple_transform(imgs, ratio[0], ratio[1], 2)


def custom_transform(imgs: torch, matrix: torch) -> torch.Tensor:
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1] == 3  # using homogeneous coordinate
    return _transform.custom_transform(imgs, matrix)


def apply_transform(imgs: torch.Tensor, trans: Transform) -> torch.Tensor:
    imgs = scale(imgs, trans.ratio)
    imgs = rotate(imgs, trans.angle)
    imgs = translate(imgs, trans.offset)
    return imgs


if __name__ == "__main__":
    from utils.imageIO import *

    img = load_raw('../imgs/barbara.raw', 256, 256, 1)
    a = translate(img, (64, 0))
    b = rotate(img, 3.1415926/4)
    c = scale(img, (4, 4))
    compare_imgs([img, a, b, c])
    plt.show()
