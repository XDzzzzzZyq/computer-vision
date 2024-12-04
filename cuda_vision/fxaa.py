import torch
from cuda_vision.__compile import load_src
from cuda_vision.convert import to_grayscale
from cuda_vision.detect import gradient_conv

_fxaa = load_src("fxaa")


def mark_edge(img: torch.Tensor, thres_min, thres_max) -> torch.Tensor:
    gray = to_grayscale(img)
    grad_x, grad_y = gradient_conv(img, 'sobel')
    return _fxaa.mark_edge(gray, grad_x, grad_y, thres_min, thres_max)


def performance(img: torch.Tensor, thres_min, thres_max, r) -> torch.Tensor:
    edge = mark_edge(img, thres_min, thres_max)
    return _fxaa.resample(img, edge, r)


def quality(img: torch.Tensor, thres_min, thres_max) -> torch.Tensor:
    pass


def console(img: torch.Tensor, thres_min, thres_max) -> torch.Tensor:
    pass


if __name__ == "__main__":
    from utils.imageIO import *

    def slice_target(tar):
        i1 = tar[:, :, 650:750, 550:650]
        i2 = tar[:, :, 1100:1200, 550:650]
        i3 = tar[:, :, 800:900, 550:650]
        i4 = tar[:, :, 850:950, 350:450]
        return [i1, i2, i3, i4]

    raw = load_png('../fxaa/imgs/alias.png')
    compare_imgs(slice_target(raw))
    print(raw.shape)
    edge = performance(raw, 20.0, 0.125, r=0.5)
    print(edge)
    compare_imgs(slice_target(edge))

    gt = load_png('../fxaa/imgs/filtered.png')
    compare_imgs(slice_target(gt))
    compare_imgs([raw, edge, gt], size=(16, 9))
    plt.show()
