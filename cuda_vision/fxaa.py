import torch
from cuda_vision.__compile import load_src
from cuda_vision.convert import to_grayscale
from cuda_vision.detect import gradient_conv

_fxaa = load_src("fxaa")


def mark_edge(img: torch.Tensor, thres_min, thres_max) -> torch.Tensor:
    gray = to_grayscale(img)
    grad_x, grad_y = gradient_conv(img, 'sobel')
    return _fxaa.mark_edge(gray, grad_x, grad_y, thres_min, thres_max)


def performance(img: torch.Tensor, thres_min, thres_max, r=1.0, hdr=False, smooth=False) -> (torch.Tensor, torch.Tensor):
    edge = mark_edge(img, thres_min, thres_max)
    if smooth:
        edge = _fxaa.smooth_offset(edge, 20)
    if hdr:
        img = img/256
        img = img/(1-img)
        print(img.min(), img.max())
    filtered = _fxaa.resample(img, edge, r)
    if hdr:
        filtered = filtered/(1+filtered)*256
        print(filtered.min(), filtered.max())
    return filtered, edge


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
    fxaa, edge = performance(raw, 20.0, 0.125, r=0.5)
    compare_imgs(slice_target(fxaa))
    compare_imgs(slice_target(edge[:, 0:1]*1000), range=None)
    compare_imgs(slice_target(edge[:, 1:2]*1000), range=None)
    fxaa_hdr, _ = performance(raw, 20.0, 0.125, r=0.5, hdr=True)
    compare_imgs(slice_target(fxaa_hdr))

    gt = load_png('../fxaa/imgs/filtered.png')
    compare_imgs(slice_target(gt))
    compare_imgs([raw, fxaa_hdr, gt], size=(16, 9))
    plt.show()
    save_png('../fxaa/imgs/fxaa_fast.png', fxaa)
