import torch
from cuda_vision.__compile import load_src
from cuda_vision.convert import to_grayscale
from cuda_vision.detect import gradient_conv

_fxaa = load_src("fxaa")


def mark_edge(gray: torch.Tensor, thres_min, thres_max, type='pixel_diff') -> torch.Tensor:
    grad_x, grad_y = gradient_conv(gray, type)
    return _fxaa.mark_edge(gray, grad_x, grad_y, thres_min, thres_max)


def performance(img: torch.Tensor, thres_min, thres_max, r=1.0, hdr=False, smooth=False, mode=0, type='pixel_diff') -> (torch.Tensor, torch.Tensor):
    gray = img.clone() if img.shape[1] == 1 else to_grayscale(img.clone())
    edge = mark_edge(gray, thres_min, thres_max, type)

    if smooth:
        edge = _fxaa.smooth_offset(gray, edge, 50, mode)
        r = 1.0

    if hdr:
        img = img/256
        img = img/(1-img)

    filtered = _fxaa.resample(img, edge, r)
    if hdr:
        filtered = filtered/(1+filtered)*256

    return filtered, gray, edge


if __name__ == "__main__":
    from utils.imageIO import *

    if False:
        def slice_target(tar):
            i1 = tar[:, :, 650:750, 550:650]
            i2 = tar[:, :, 1100:1200, 550:650]
            i3 = tar[:, :, 800:900, 550:650]
            i4 = tar[:, :, 850:950, 350:450]
            i5 = tar[:, :, 1300:1400, 550:650]
            return [i1, i2, i3, i4, i5]

        raw = load_png('../fxaa/imgs/alias.png')
        compare_imgs(slice_target(raw))
        fxaa, _, edge = performance(raw, 20.0, 0.125, r=0.5, hdr=True)
        compare_imgs(slice_target(fxaa))
        fxaa_s0, _, smooth0 = performance(raw, 20.0, 0.125, hdr=True, smooth=True, mode=0)
        compare_imgs(slice_target(fxaa_s0))
        fxaa_s1, _, smooth1 = performance(raw, 20.0, 0.125, hdr=True, smooth=True, mode=1)

        compare_imgs(slice_target(fxaa_s1))
        compare_imgs_grid(slice_target(edge[:, 0:1]*1000)+slice_target(edge[:, 1:2]*1000)+slice_target(smooth1[:, 0:1]*1000)+slice_target(smooth1[:, 1:2]*1000), range=None, shape=(4, 5))

        gt = load_png('../fxaa/imgs/filtered.png')
        compare_imgs(slice_target(gt))
        plt.show()
        save_png('../fxaa/imgs/fxaa_fast.png', fxaa_s1)

        error = torch.nn.MSELoss()
        print(error(gt, raw), error(gt, fxaa), error(gt, fxaa_s1))

    else:
        from cuda_vision.convert import invert
        img = load_raw('../imgs/patterns.raw', 256, 256, 1)
        img = invert(img)

        fxaa, _, edge = performance(img, 20.0, 0.125, r=0.5)
        fxaa_s0, _, smooth0 = performance(img, 20.0, 0.125, smooth=True, mode=0, type='sobel')
        fxaa_s1, _, smooth1 = performance(img, 20.0, 0.125, smooth=True, mode=1)

        compare_imgs([edge[:, 0:1], smooth0[:, 0:1], smooth1[:, 0:1]])
        compare_imgs([edge[:, 1:2], smooth0[:, 1:2], smooth1[:, 1:2]])
        plt.show()
