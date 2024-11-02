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
static __global__ void get_gaussian_kernel(
    scalar_t* kernel,
    float std, int size
) {
    int idx = threadIdx.x;
}

template <typename scalar_t>
static __global__ void semi_conv_gray_kernel(
    scalar_t* result,
    const scalar_t* gray,
    const float* semi_kernel,
    int size, int pad
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x;
    int w = gridDim.y - 2 * (pad - size);

    scalar_t r = 0.0;
    for(int i = -size; i<size+1; i++){
        int im_x = x+size-pad+i;
        int im_y = y+size-pad;

        float weight = semi_kernel[i+size];
        scalar_t p = (im_x < 0 || im_x >= w) ? 0 : get_value(gray, im_x, im_y, h, w);
        r += p * weight;
    }

    set_value(result, r, y, x); // transpose
}

template <typename scalar_t>
static __global__ void semi_conv_kernel(
    scalar_t* result,
    const scalar_t* image,
    const float* semi_kernel,
    int size, int pad
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> r;
    for(int i = -size; i<size+1; i++){
        float w = semi_kernel[i+size];
        pixel<scalar_t> p = get_pixel(image, x, y);
    }

    get_pixel(image, x, y);
}

// C++ API

void uniform_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int size, int pad
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    int e = pad - size;

    float* kernel = make_array<float>(2*size+1, 1.0/(2.0*size+1.0));

    torch::Tensor temp = torch::empty({b, c, w+2*e, h}).to(image.device()); // transpose
    dim3 grid_size1(h,     w+2*e, 1);
    dim3 grid_size2(h+2*e, w+2*e, 1);

    if(c == 3){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_kernel", [&] {
            semi_conv_kernel<scalar_t><<<grid_size1, b, 0, stream>>>(
                temp.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_kernel", [&] {
            semi_conv_kernel<scalar_t><<<grid_size2, b, 0, stream>>>(
                result.data_ptr<scalar_t>(),
                temp.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
    }else if (c == 1){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_gray_kernel", [&] {
            semi_conv_gray_kernel<scalar_t><<<grid_size1, b, 0, stream>>>(
                temp.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_gray_kernel", [&] {
            semi_conv_gray_kernel<scalar_t><<<grid_size2, b, 0, stream>>>(
                result.data_ptr<scalar_t>(),
                temp.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
    }
}

void gaussian_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float std, int size, int pad
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, 1);
}