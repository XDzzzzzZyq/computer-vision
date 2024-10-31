#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

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