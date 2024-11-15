import torch
from cuda_vision.__compile import load_src

_convert = load_src("convert")


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    return _convert.to_grayscale(img)


def invert(img: torch.Tensor) -> torch.Tensor:
    return _convert.invert(img)


def hash(img: torch.Tensor, step=1, seed=123) -> torch.Tensor:
    result = img
    for i in range(step):
        result = _convert.hash(result, seed)
    return result


def threshold(img: torch, threshold) -> torch.Tensor:
    return _convert.threshold(img, threshold)


def random_threshold(img: torch, type, std=127.5, seed=123, step=2) -> torch.Tensor:
    if type == 'uniform':
        noise = hash(img, step=step, seed=seed)
        noise = (noise - 127.5) * std * (12 ** 0.5) / 255.0 + 127.5
    elif type == 'gaussian':
        noise = torch.randn_like(img) * std + 127.5
        noise = torch.clamp(noise, 0.0, 255.0)
    else:
        noise = torch.randn_like(img)
    return _convert.random_threshold(img, noise)


def matrix_dither(img: torch.Tensor, n) -> torch.Tensor:
    from cuda_vision.__kernels import get_dither_matrix
    assert n % 2 == 0
    assert img.shape[2] % n == 0 and img.shape[3] % n == 0
    index = get_dither_matrix(n).to(img.device).to(img.dtype)
    return _convert.matrix_dither(img, index)
