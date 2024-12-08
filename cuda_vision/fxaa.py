import torch
from cuda_vision.__compile import load_src
from cuda_vision.convert import to_grayscale
from cuda_vision.detect import gradient_conv

_fxaa = load_src("fxaa")


def mark_edge(gray: torch.Tensor, thres_min, thres_max) -> torch.Tensor:
    grad_x, grad_y = gradient_conv(gray, 'pixel_diff')
    return _fxaa.mark_edge(gray, grad_x, grad_y, thres_min, thres_max)


def performance(img: torch.Tensor, thres_min, thres_max, r=1.0, hdr=False, smooth=False, mode=0) -> (torch.Tensor, torch.Tensor):
    gray = img.clone() if img.shape[1] == 1 else to_grayscale(img)
    edge = mark_edge(gray, thres_min, thres_max)

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

    if True:
        def slice_target(tar):
            i1 = tar[:, :, 650:750, 550:650]
            i2 = tar[:, :, 1100:1200, 550:650]
            i3 = tar[:, :, 800:900, 550:650]
            i4 = tar[:, :, 850:950, 350:450]
            i5 = tar[:, :, 1300:1400, 550:650]
            return [i1, i2, i3, i4, i5]

        raw = load_png('../fxaa/imgs/alias.png')
        compare_imgs(slice_target(raw))
        fxaa, gray, edge = performance(raw, 20.0, 0.125, r=0.5, hdr=True)
        compare_imgs(slice_target(fxaa))
        fxaa_s, gray, smooth = performance(raw, 20.0, 0.125, hdr=True, smooth=True, mode=1)
        compare_imgs(slice_target(fxaa_s))
        compare_imgs_grid(slice_target(edge[:, 0:1]*1000)+slice_target(edge[:, 1:2]*1000)+slice_target(smooth[:, 0:1]*1000)+slice_target(smooth[:, 1:2]*1000), range=None, shape=(4, 5))

        gt = load_png('../fxaa/imgs/filtered.png')
        compare_imgs(slice_target(gt))
        plt.show()
        save_png('../fxaa/imgs/fxaa_fast.png', fxaa_s)

        error = torch.nn.MSELoss()
        print(error(gt, raw), error(gt, fxaa), error(gt, fxaa_s))

    else:
        from cuda_vision.convert import invert
        raw = load_raw('../imgs/patterns.raw', 256, 256, 1)
        raw = invert(raw)

        fxaa, gray, edge = performance(raw, 2.0, 0.125, r=0.5)
        smooth = _fxaa.smooth_offset(gray, edge, 20)

        compare_imgs_grid([edge[:, 0:1]*1000, edge[:, 1:2]*1000, smooth[:, 0:1]*1000, smooth[:, 1:2]*1000], range=None, shape=(2,2))

        fxaa_s, gray, edge = performance(raw, 20.0, 0.125, smooth=True)
        compare_imgs([raw, fxaa, fxaa_s])
        plt.show()