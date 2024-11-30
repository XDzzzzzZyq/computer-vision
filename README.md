# CUDA Vision
Open-source Computer Vision Library implemented by CUDA. https://github.com/XDzzzzzZyq/computer-vision

## Introduction

This is an open-source computer vision library developed in Python, utilizing the PyTorch framework and optimized with CUDA for accelerated processing on GPUs. Designed to be flexible and powerful, the library provides a range of advanced computer vision tools and algorithms, enabling high-performance image processing and deep learning applications. With CUDA integration, it harnesses the computational power of GPUs, making it suitable for handling large-scale datasets and real-time applications. 

### Image Format
This library treats images as tensors that strictly follows the shape ```(B,C,H,W)```. Where ```B``` is the batch size, 
```C``` is the number of channels, ```H,W``` are the height and width of images correspondingly.

Two storage methods are supported, unsigned bytes ranging within ```[0,255]``` or standard ```float32```.

### CUDA Vision

All CUDA operators will be complied during run-time. Take grays-cale conversion as an example:

```python
from cuda_vision.convert import to_grayscale
from cuda_vision.enhance import uniform_equalize
from cuda_vision.filters import bilateral_filter
from utils.imageIO import load_raw, compare_imgs

img = load_raw('imgs/building2_color.raw', 256, 256, 3)
gray1 = to_grayscale(img)
gray2 = uniform_equalize(gray1, 64)
gray3 = bilateral_filter(gray2, 10.0, 50.0, 20, 20)
compare_imgs([img, gray1, gray2, gray3])
```

The following functionalities are supported:

`convert`

- Grayscale Conversion
- Grayscale Inversion
- Binarization
  - Thresholding
  - Random Thresholding
  - Dithering & Error Diffusion

`combine`

- Making watermark
- Float/Logic Arithmetics
- Clamp

`enhance`

- Min-Max Scaling
- Histogram Equalization

`filters`

- Convolution
- Blurring
- Pattern Matching
- Morphology Operations: ```shrink```, ```thin```, ```skeletonize```, ...

`detect`

- Edge detection (Gradient method & Laplacian method)

`transform`

- Elementary Transformation
- Homogeneous Matrix Transformation
- Disk Warping

`feature`

- Momentum Features
- Law's Features
- Min / Max / Median
- Geometry (Metric) Features
- Topology (Euler) Features

### Utils

For imageIO and visualization, tools are provided in the ```utils```.

#### Load Image
For ```.raw``` files, it should indicate the ```width```, ```height```, and ```channel``` of current image.
```python
from utils.imageIO import load_raw
ori = load_raw('imgs/rose_color.raw',       256, 256, 3)
noi = load_raw('imgs/rose_color_noise.raw', 256, 256, 3)
```

#### Show Image & Histogram
```python
from utils.imageIO import show_img, compare_imgs, show_hist, compare_hist
show_img(ori)
compare_imgs([ori, noi, denoised])
show_hist(ori)
compare_hist([ori, noi, denoised])
compare_hist([ori, noi, denoised], accu=True)
```



## Required Environment

- Matplotlib
- Pytorch (CUDA)
- SciPy
- A Nvidia Graphics Card