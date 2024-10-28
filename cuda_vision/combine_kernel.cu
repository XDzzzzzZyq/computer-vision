// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
struct pixel{
    scalar_t r;
    scalar_t g;
    scalar_t b;

    __device__ pixel<scalar_t> operator*(float fact){
        return pixel<scalar_t>{scalar_t(r*fact), scalar_t(g*fact), scalar_t(b*fact)};
    }
    __device__ pixel<scalar_t> operator+(const pixel<scalar_t>& tar){
        return pixel<scalar_t>{r+tar.r, g+tar.g, b+tar.b};
    }
    __device__ pixel<scalar_t> operator-(const pixel<scalar_t>& tar){
        return *self + tar * -1.0;
    }
    __device__ __host__ scalar_t grayscale(){
        return 0.299*r + 0.587*g + 0.114*b;
    }
};

template <typename scalar_t>
static __device__ pixel<scalar_t> get_grayscale(
    const scalar_t* gray,
    int x, int y
){
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    return gray[batch + y * gridDim.x + x];
}

template <typename scalar_t>
static __device__ pixel<scalar_t> get_pixel(
    const scalar_t* image,
    int x, int y, int h=0, int w=0
){
    h = h == 0 ? gridDim.x : h;
    w = w == 0 ? gridDim.y : w;
    int chane_off = h * w;
    int batch_off = threadIdx.x * chane_off * 3;
    int loc = y * h + x;
    scalar_t r = image[batch_off + loc];
    scalar_t g = image[batch_off + loc + chane_off];
    scalar_t b = image[batch_off + loc + chane_off*2];
    return pixel<scalar_t>{r, g, b};
}

template <typename scalar_t>
static __device__ void set_pixel(
    scalar_t* image,
    pixel<scalar_t> pixel,
    int x, int y, int h=0, int w=0
){
    h = h == 0 ? gridDim.x : h;
    w = w == 0 ? gridDim.y : w;
    int chane_off = h * w;
    int batch_off = threadIdx.x * chane_off * 3;
    int loc = y * h + x;
    image[batch_off + loc] = pixel.r;
    image[batch_off + loc + chane_off] = pixel.g;
    image[batch_off + loc + chane_off*2] = pixel.b;
}

template <typename scalar_t>
static __global__ void watermark_kernel(
    scalar_t* result,
    const scalar_t* image,
    const scalar_t* mark,
    int offset_x, int offset_y,
    int img_h, int img_w
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> pixel_mk = get_pixel(mark, x, y);
    scalar_t grayscale_mk = pixel_mk.grayscale();

    if(grayscale_mk < 245.0)
        set_pixel(result, pixel_mk, x+offset_x, y+offset_y, img_h, img_w);
}

// C++ API

void watermark_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& mark,
    int offset_x, int offset_y
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = mark.size(2);
    int w = mark.size(3);
    dim3 grid_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "watermark_kernel", [&] {
        watermark_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            mark.data_ptr<scalar_t>(),
            offset_x, offset_y,
            image.size(2), image.size(3)
        );
    });
}