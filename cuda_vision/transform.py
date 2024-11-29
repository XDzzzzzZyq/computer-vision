import matplotlib.pyplot as plt
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


def island_segment(imgs: torch.Tensor, pad=3) -> list[torch.Tensor]:
    from scipy.ndimage import label
    from convert import threshold
    B, _, H, W = imgs.shape
    labeled_image, num_features = label(imgs.cpu())
    labeled_image = torch.from_numpy(labeled_image).float()
    islands = [(labeled_image == label).float().to(imgs.device) for label in range(1, num_features+1)]

    def get_aabb(island):
        _, _, cols, rows = island.nonzero(as_tuple=True)
        y_min, y_max = rows.min().item(), rows.max().item()
        x_min, x_max = cols.min().item(), cols.max().item()
        return x_min-pad, x_max+pad+1, y_min-pad, y_max+pad+1

    coord = [get_aabb(island) for island in islands]
    segments = []
    for c, island in zip(coord, islands):
        x_min, x_max, y_min, y_max = c
        offset = (W/2 - (x_max+x_min)/2, H/2 - (y_max+y_min)/2)
        ratio = (W/(x_max-x_min), H/(y_max-y_min))
        island = translate(island, offset=offset)
        island = scale(island, ratio=ratio)
        island = threshold(island, 0.5)
        segments.append(island)
    return segments


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


def disk_warp(img: torch.Tensor, inverse=False) -> torch.Tensor:
    return _transform.disk_warp(img, inverse)


if __name__ == "__main__":
    from utils.imageIO import *
    from convert import invert, threshold
    import math

    train = load_raw('../imgs/training.raw', 256, 256, 3)[:,0:1].contiguous()
    train = invert(train)
    seg = island_segment(train)
    compare_imgs_grid(seg, shape=(3, 4), range=None)
    plt.show()
