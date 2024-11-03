# CUDA Vision
Open-source Computer Vision Library implemented by CUDA.

## Introduction

This is an open-source computer vision library developed in Python, utilizing the PyTorch framework and optimized with CUDA for accelerated processing on GPUs. Designed to be flexible and powerful, the library provides a range of advanced computer vision tools and algorithms, enabling high-performance image processing and deep learning applications. With CUDA integration, it harnesses the computational power of GPUs, making it suitable for handling large-scale datasets and real-time applications. 

### Image Format
This library treats images as tensors that strictly follows the shape ```(B,C,H,W)```. Where ```B``` is the batch size, 
```C``` is channel numbers, ```H,W``` are the height and width of images correspondingly.

Two storage methods are supported, unsigned bytes ranging within ```[0,255]``` or standard ```float32```.

### CUDA Vision

All CUDA operators will be complied during run-time. Take grays-cale conversion as an example:

```python
from cuda_vision import to_grayscale

img = load_raw('imgs/building2_color.raw', 256, 256, 3)
gray = to_grayscale(img)
compare_imgs([img, gray])
```

The following functionalities are supported:
- `to_grayscale(img: torch.Tensor)` 

- `make_watermark(img: torch.Tensor, mark: torch.Tensor, offset=(0, 0))`

- `invert(img: torch.Tensor)`

- `minmax_scale(img: torch.Tensor)`

- `uniform_equalize(img: torch.Tensor, k)`

- `uniform_conv(img: torch.Tensor, size, pad)`

- `gaussian_conv(img: torch.Tensor, std, size, pad)`

- `median_filter(img: torch.Tensor, size, pad)`

- `pseudo_median_filter(img: torch.Tensor, size, pad)`

- `bilateral_filter(img: torch.Tensor, std_s, std_i, size, pad)`

- `add`, `subtract`, `multiply`, `divide`

### Utils

For imageIO and visualization, tools are provided in the ```utils```.

#### Load Image
For ```.raw``` files, it should indicate the ```width```, ```height```, and ```channel``` of current image.
```python
ori = load_raw('imgs/rose_color.raw',       256, 256, 3)
noi = load_raw('imgs/rose_color_noise.raw', 256, 256, 3)
```

#### Show Image & Histogram
```python
show_img(ori)
compare_imgs([ori, noi, denoised])
show_hist(ori)
compare_hist([ori, noi, denoised])
compare_hist([ori, noi, denoised], accu=True)
```



## Required Environment

- Matplotlib
- Pytorch (CUDA)
- A Nvidia Graphics Card