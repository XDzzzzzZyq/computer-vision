#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void overlay_kernel(
    scalar_t* result,
    const scalar_t* a,
    const scalar_t* b,
    int code
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> pix_a = get_pixel(a, x, y);
    pixel<scalar_t> pix_b = get_pixel(b, x, y);
    pixel<scalar_t> pix_c;

    switch(code){
    case 0: pix_c = pix_a + pix_b; break;
    case 1: pix_c = pix_a - pix_b; break;
    case 2: pix_c = pix_a * pix_b; break;
    case 3: pix_c = pix_a / pix_b; break;
    }
    set_pixel(result, pix_c, x, y);
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

template <typename scalar_t>
static __global__ void watermark_gray_kernel(
    scalar_t* result,
    const scalar_t* image,
    const scalar_t* mark,
    int offset_x, int offset_y,
    int img_h, int img_w
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    scalar_t grayscale_mk = get_value(mark, x, y);

    if(grayscale_mk < 245.0)
        set_value(result, grayscale_mk, x+offset_x, y+offset_y, img_h, img_w);
}

// C++ API

void overlay_op(
    torch::Tensor& result,
    const torch::Tensor& im_a,
    const torch::Tensor& im_b,
    int code
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = im_a.size(0);
    int h = im_a.size(2);
    int w = im_a.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(im_a.scalar_type(), "overlay_kernel", [&] {
        overlay_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            im_a.data_ptr<scalar_t>(),
            im_b.data_ptr<scalar_t>(),
            code
        );
    });
}

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
    int c = image.size(1);
    int h = mark.size(2);
    int w = mark.size(3);
    dim3 grid_size(h, w, b);

    if (c == 3){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "watermark_kernel", [&] {
            watermark_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
                result.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                mark.data_ptr<scalar_t>(),
                offset_x, offset_y,
                image.size(2), image.size(3)
            );
        });
    }else if (c == 1){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "watermark_gray_kernel", [&] {
            watermark_gray_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
                result.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                mark.data_ptr<scalar_t>(),
                offset_x, offset_y,
                image.size(2), image.size(3)
            );
        });
    }
}
