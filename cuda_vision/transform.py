import torch
from typing import overload
from cuda_vision.__compile import load_src

_transform = load_src("transform")


class Transform(object):
    def __init__(self, offset=(0.0,0.0), angle=0.0, ratio=(1.0,1.0)):
        self.offset = offset
        self.angle = angle
        self.ratio = ratio if isinstance(ratio, tuple) else (ratio, ratio)


def translate(imgs: torch.Tensor, offset: tuple[float, float]) -> torch.Tensor:
    return _transform.simple_transform(imgs, offset[1], offset[0], 0)


def rotate(imgs: torch.Tensor, angle: float) -> torch.Tensor:
    return _transform.simple_transform(imgs, angle, 0.0, 1)


def scale(imgs: torch.Tensor, ratio: tuple[float, float] or float) -> torch.Tensor:
    if isinstance(ratio, tuple):
        if ratio[0] == 0.0 or ratio[1] == 0.0:
            return torch.zeros_like(imgs)
        return _transform.simple_transform(imgs, ratio[1], ratio[0], 2)
    elif isinstance(ratio, float):
        if ratio == 0.0:
            return torch.zeros_like(imgs)
        return _transform.simple_transform(imgs, ratio, ratio, 2)
    return imgs


def custom_transform(imgs: torch.Tensor, trans: torch.Tensor or Transform) -> torch.Tensor:

    if isinstance(trans, Transform):
        import math
        offset = trans.offset
        angle = trans.angle
        ratio = trans.ratio
        matrx = torch.tensor([[ratio[0] * math.cos(angle),-ratio[1] * math.sin(angle), offset[0]],
                              [ratio[0] * math.sin(angle), ratio[1] * math.cos(angle), offset[1]],
                              [0.0, 0.0, 1.0]]).to(imgs.device).float()
        return custom_transform(imgs, matrx)
    elif isinstance(trans, torch.Tensor):
        assert trans.ndim == 2
        assert trans.shape[0] == trans.shape[1] == 3  # using homogeneous coordinate
        return _transform.custom_transform(imgs, trans)

    return imgs


if __name__ == "__main__":
    from utils.imageIO import *
    import math

    img = load_raw('../imgs/barbara.raw', 256, 256, 1)
    imgs = []
    for t in range(21):
        offset = math.sqrt(2) * t
        angle = 5 * t / 360 * (2 * math.pi)
        ratio = 0.97 ** t
        trans = Transform(offset=(offset, offset), angle=angle, ratio=ratio)
        imgs.append(custom_transform(img, trans))

    compare_imgs([imgs[0], imgs[5], imgs[20]])
    plt.show()
