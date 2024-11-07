#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

template <typename scalar_t>
__host__ __device__ scalar_t _math(scalar_t a, scalar_t b, int code) {
    scalar_t r = 0;
    switch(code){
    case 0: r = a + a; break;
    case 1: r = a - a; break;
    case 2: r = a * a; break;
    case 3: r = a / a; break;
    case 4: r = sqrtf(a*a + b*b);
    }
    return r;
}

template <typename scalar_t>
static __global__ void overlay_kernel(
    scalar_t* result,
    const scalar_t* a,
    const scalar_t* b,
    int code, bool rgb
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    if(rgb){
        pixel<scalar_t> pix_a = get_pixel(a, x, y);
        pixel<scalar_t> pix_b = get_pixel(b, x, y);
        pixel<scalar_t> pix_c;

        pix_c.r = _math(pix_a.r, pix_b.r, code);
        pix_c.g = _math(pix_a.g, pix_b.g, code);
        pix_c.b = _math(pix_a.b, pix_b.b, code);

        set_pixel(result, pix_c, x, y);
    }else{
        scalar_t pix_a = get_value(a, x, y);
        scalar_t pix_b = get_value(b, x, y);
        scalar_t pix_c = _math(pix_a, pix_b, code);

        set_value(result, pix_c, x, y);
    }
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
    int c = im_a.size(1);
    int h = im_a.size(2);
    int w = im_a.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(im_a.scalar_type(), "overlay_kernel", [&] {
        overlay_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            im_a.data_ptr<scalar_t>(),
            im_b.data_ptr<scalar_t>(),
            code, c==3
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
