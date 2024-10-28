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
    int x, int y
){
    int chane_off = gridDim.x * gridDim.y;
    int batch_off = threadIdx.x * chane_off * 3;
    int loc = y * gridDim.x + x;
    scalar_t r = image[batch_off + loc];
    scalar_t g = image[batch_off + loc + chane_off];
    scalar_t b = image[batch_off + loc + chane_off*2];
    return pixel<scalar_t>{r, g, b};
}

template <typename scalar_t>
static __global__ void to_grayscale_kernel(
    scalar_t* gray,
    const scalar_t* image
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> pixel = get_pixel(image, x, y);
    scalar_t grayscale = 0.299*pixel.r + 0.587*pixel.g + 0.114*pixel.b;

    int batch = threadIdx.x * gridDim.x * gridDim.y;
    gray[batch + y * gridDim.x + x] = grayscale;
}

// C++ API

void to_grayscale_op(
    torch::Tensor& gray,
    const torch::Tensor& image
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "to_grayscale_kernel", [&] {
        to_grayscale_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            gray.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>()
        );
    });
}