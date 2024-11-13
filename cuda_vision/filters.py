import torch
from cuda_vision.__compile import load_src

_filter = load_src("filter")


def uniform_conv(img: torch.Tensor, size, pad) -> torch.Tensor:
    return _filter.uniform_conv(img, size, pad)


def gaussian_conv(img: torch.Tensor, std, size, pad) -> torch.Tensor:
    assert std > 0.0, "std should be positive"
    return _filter.gaussian_conv(img, std, size, pad)


def custom_conv(img: torch.Tensor, kernel: torch.Tensor, pad) -> torch.Tensor:
    return _filter.custom_conv(img, kernel, pad)


def median_filter(img: torch.Tensor, size, pad) -> torch.Tensor:
    return _filter.median_filter(img, size, pad, False)


def pseudo_median_filter(img: torch.Tensor, size, pad) -> torch.Tensor:
    return _filter.median_filter(img, size, pad, True)


def bilateral_filter(img: torch.Tensor, std_s, std_i, size, pad) -> torch.Tensor:
    return _filter.bilateral_filter(img, std_s, std_i, size, pad)


def conditional_match(img: torch.Tensor, type) -> torch.Tensor:
    from cuda_vision.__kernels import get_conditional_patterns
    patterns = get_conditional_patterns(type).to(img.dtype)
    return _filter.conditional_match(img, patterns)
