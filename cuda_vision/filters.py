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


def pattern_match(img: torch.Tensor, type, cond=True) -> torch.Tensor:
    from cuda_vision.__kernels import get_conditional_patterns, get_unconditional_patterns
    if cond:
        patterns = get_conditional_patterns(type).to(img.dtype).to(img.device)
    else:
        patterns = get_unconditional_patterns(type).to(img.dtype).to(img.device)
    return _filter.pattern_match(img, patterns, cond)


def morphology(img: torch.Tensor, type) -> torch.Tensor:

    '''
    Shrinks/Thin/Skeletonizes/Erode/Dilate the Binary images.
    :param img: Target Images
    :param type:Type of process
    :return: Processed Images
    '''

    if type in ['S', 's', 'shrink', 'T', 't', 'thin', 'K', 'k', 'skeletonize']:
        from cuda_vision import combine, convert
        mark = pattern_match(img,  type=type, cond=True)
        prsv = pattern_match(mark, type=type, cond=False)

        # G = X \cap (\not M \cup P)
        m_cp = combine.lor(convert.invert(mark), prsv)
        g = combine.land(img, m_cp)
    elif type in ['E', 'e', 'erode']:
        patterns = torch.ones(1, 3, 3).to(img.dtype).to(img.device)
        g = _filter.pattern_match(img, patterns, True)
    elif type in ['D', 'd', 'dilate']:
        from cuda_vision import convert
        g = convert.invert(img)
        g = morphology(g, 'e')
        g = convert.invert(g)
    else:
        g = img

    return g
