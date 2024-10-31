#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"
#include "arrayUtils.cuh"

template <typename scalar_t>
static __global__ void minmax_kernel(
    int* batched_min, int* batched_max,
    const scalar_t* image
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int b = threadIdx.x;

    scalar_t gray = get_value(image, x, y);
    atomicMin(&batched_min[b], int(gray));
    atomicMax(&batched_max[b], int(gray));
}

template <typename scalar_t>
static __global__ void minmax_scale_kernel(
    scalar_t* result,
    const scalar_t* image,
    const int* batched_min, const int* batched_max
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int b = threadIdx.x;

    scalar_t gray = get_value(image, x, y);
    scalar_t scaled = (gray - batched_min[b])/(batched_max[b] - batched_min[b]) * 255.;
    set_value(result, scaled, x, y);
}

// C++ API

void minmax_scale_op(
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

    int* batched_min = make_array(b, INT_MAX);
    int* batched_max = make_array(b, INT_MIN);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "minmax_kernel", [&] {
        minmax_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            batched_min, batched_max,
            image.data_ptr<scalar_t>()
        );
    });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "minmax_scale_kernel", [&] {
        minmax_scale_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            batched_min, batched_max
        );
    });

    cudaFree(batched_min);
    cudaFree(batched_max);
}