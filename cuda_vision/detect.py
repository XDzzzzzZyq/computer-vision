import torch

import __kernels
import filters


def gradient_conv(img: torch.Tensor, type):
    row_kernel, col_kernel = __kernels.get_gradient_kernel(type)
    grad_x = filters.custom_conv(img, row_kernel, 1)
    grad_y = filters.custom_conv(img, col_kernel, 1)
    return grad_x, grad_y


def gradient_edge_detect(img: torch.Tensor, threshold, type):
    import combine, convert
    grad_x, grad_y = gradient_conv(img, type)
    grad = combine.length(grad_x, grad_y)
    edge = convert.binarize(grad, threshold)
    return edge
