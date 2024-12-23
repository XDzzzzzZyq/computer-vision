import torch

import cuda_vision.__kernels as __kernels
from cuda_vision.__compile import load_src

_detect = load_src("detect")


def gradient_conv(img: torch.Tensor, type):
    from cuda_vision import filters
    row_kernel, col_kernel = __kernels.get_gradient_kernel(type)
    grad_x = filters.custom_conv(img, row_kernel, 1)
    grad_y = filters.custom_conv(img, col_kernel, 1)
    return grad_x, grad_y


def gradient_edge_detect(img: torch.Tensor, threshold, type):
    from cuda_vision import combine, convert
    grad_x, grad_y = gradient_conv(img, type)
    grad = combine.length(grad_x, grad_y)
    edge = convert.threshold(grad, threshold)
    return edge


def laplacian_conv(img: torch.Tensor, type):
    from cuda_vision import filters
    ker = __kernels.get_laplacian_kernel(type)
    pad = ker.shape[0] // 2
    lap = filters.custom_conv(img, ker, pad=pad)
    return lap


def laplacian_edge_detect(img: torch.Tensor, type, threshold, mode=0):
    lap = laplacian_conv(img, type)
    edge = _detect.zero_crossing(lap, threshold, mode)
    return edge
