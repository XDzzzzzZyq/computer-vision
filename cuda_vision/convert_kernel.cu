#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

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

template <typename scalar_t>
static __global__ void invert_kernel(
    scalar_t* result,
    const scalar_t* image
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> pixel = get_pixel(image, x, y);
    pixel = (pixel - 255.) * -1.;

    set_pixel(result, pixel, x, y);
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

void invert_op(
    torch::Tensor& result,
    const torch::Tensor& image
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "invert_kernel", [&] {
        invert_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>()
        );
    });
}