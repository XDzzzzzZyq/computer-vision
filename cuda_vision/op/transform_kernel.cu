#include <torch/types.h>
#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void simple_transform_kernel(
    scalar_t* result,
    const scalar_t* image,
    float arg1, float arg2,
    int mode,
    int w_old, int h_old
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int w = gridDim.x;
    int h = gridDim.y;

    float u = (float(x)+0.5)/float(w) - 0.5;
    float v = (float(y)+0.5)/float(h) - 0.5;
    float _u, _v;

    switch(mode){
    case 0:
        u -= arg1/float(w_old);
        v -= arg2/float(h_old);
        break;
    case 1:
        _u = u*cos(arg1) - v*sin(arg1);
        _v = u*sin(arg1) + v*cos(arg1);
        u = _u;
        v = _v;
        break;
    case 2:
        u = u/arg1;
        v = v/arg2;
        break;
    case 3:
        _u = u + arg1*v;
        _v = v + arg2*u;
        u = _u;
        v = _v;
        break;
    }
    scalar_t value = sample_value(image, u+0.5, v+0.5, w_old, h_old);
    set_value(result, value, x, y);
}

template <typename scalar_t>
static __global__ void custom_transform_kernel(
    scalar_t* result,
    const scalar_t* image,
    const float* inv,
    float off_x, float off_y,
    int w_old, int h_old
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int w = gridDim.x;
    int h = gridDim.y;

    float u = (float(x)+0.5)/float(w) - 0.5 - off_x / w_old;
    float v = (float(y)+0.5)/float(h) - 0.5 - off_y / h_old;
    float _u = u * inv[0] + v * inv[1];
    float _v = u * inv[2] + v * inv[3];
    u = _u + 0.5;
    v = _v + 0.5;

    scalar_t value = sample_value(image, u, v, w_old, h_old);
    set_value(result, value, x, y);
}

template <typename scalar_t>
static __global__ void disk_warp_kernel(
    scalar_t* result,
    const scalar_t* image,
    bool inverse
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int w = gridDim.x;
    int h = gridDim.y;

    float u = (float(x)+0.5)/float(w) - 0.5;
    float v = (float(y)+0.5)/float(h) - 0.5;

    float r = sqrtf(u*u + v*v)/(abs(u) > abs(v) ? abs(u) : abs(v));
    if(inverse)
        r = 1/r;

    u = r*u + 0.5;
    v = r*v + 0.5;

    scalar_t value = sample_value(image, u, v);
    set_value(result, value, x, y);
}

// C++ API

void simple_transform_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float arg1, float arg2,
    int mode
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int w = image.size(2);
    int h = image.size(3);
    int w_new = result.size(2);
    int h_new = result.size(3);
    dim3 grid_size(w_new, h_new, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "simple_transform_kernel", [&] {
        simple_transform_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            arg1, arg2, mode,
            w, h
        );
    });
}

void custom_transform_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& matrix
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int w = image.size(2);
    int h = image.size(3);
    int h_new = result.size(2);
    int w_new = result.size(3);
    dim3 grid_size(w_new, h_new, b);

    torch::Tensor inv = matrix.index({torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)});
    inv = torch::linalg::inv(inv);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "custom_transform_kernel", [&] {
        custom_transform_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            inv.data_ptr<float>(),
            matrix.index({0, 2}).item<float>(), matrix.index({1, 2}).item<float>(),
            w, h
        );
    });
}

void disk_warp_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    bool inverse
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int w = image.size(2);
    int h = image.size(3);
    dim3 grid_size(w, h, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "disk_warp_kernel", [&] {
        disk_warp_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            inverse
        );
    });
}