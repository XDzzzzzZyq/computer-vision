#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

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

    int batch = blockIdx.z * gridDim.x * gridDim.y;
    gray[batch + y * gridDim.x + x] = grayscale;
}

template <typename scalar_t>
static __global__ void invert_kernel(
    scalar_t* result,
    const scalar_t* image,
    bool rgb
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    if(rgb){
        pixel<scalar_t> pixel = get_pixel(image, x, y);
        pixel = (pixel - 255.) * -1.;
        set_pixel(result, pixel, x, y);
    }else{
        scalar_t gray = get_value(image, x, y);
        gray = (gray - 255.) * -1.;
        set_value(result, gray, x, y);
    }
}

template <typename scalar_t>
static __global__ void hash_kernel(
    scalar_t* result,
    const scalar_t* image,
    int seed,
    int prime=982451653
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int mod = pow(10,9) + 7;

    int coor_seed = ((y * gridDim.x + x + seed) * prime) % 255;
    scalar_t gray = get_value(image, x, y);
    gray = fmod((gray + coor_seed) * prime, scalar_t(mod));
    gray/= mod/255.0;
    set_value(result, gray, x, y);
}

template <typename scalar_t>
static __global__ void threshold_kernel(
    scalar_t* result,
    const scalar_t* image,
    float threshold, bool rgb
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

        if(rgb){
        pixel<scalar_t> pixel = get_pixel(image, x, y);
        pixel.r = pixel.r < threshold ? 0.0 : 255.0;
        pixel.g = pixel.g < threshold ? 0.0 : 255.0;
        pixel.b = pixel.b < threshold ? 0.0 : 255.0;
        set_pixel(result, pixel, x, y);
    }else{
        scalar_t gray = get_value(image, x, y);
        gray = gray < threshold ? 0.0 : 255.0;
        set_value(result, gray, x, y);
    }
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
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "to_grayscale_kernel", [&] {
        to_grayscale_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
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
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "invert_kernel", [&] {
        invert_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            c == 3
        );
    });
}

void hash_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int seed
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "hash_kernel", [&] {
        hash_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            seed
        );
    });
}

void threshold_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float threshold
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "threshold_kernel", [&] {
        threshold_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            threshold, c == 3
        );
    });
}