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
static __global__ void mark_edge_kernel(
    scalar_t* result,
    const scalar_t* gray,
    const scalar_t* grad_x,
    const scalar_t* grad_y,
    float thres_min, float thres_max
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int w = gridDim.x;
    int h = gridDim.y;

    scalar_t coord_x = 0.0;
    scalar_t coord_y = 0.0;

    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int im_x = x-1 + t%3;
    int im_y = y-1 + t/3;
    buffer[t] = IS_OUT(im_x, im_y, w, h) ? 0 : get_value(gray, im_x, im_y, w, h);
    __syncthreads();

    if (t!=0)
        return;

    scalar_t gray_min = arr::min(buffer, 9);
    scalar_t gray_max = arr::max(buffer, 9);
    scalar_t gray_range = gray_max - gray_min;

    if(gray_range < max(thres_min, gray_max*thres_max)){
        set_value_channel(result, coord_x, x, y, 0, 2);
        set_value_channel(result, coord_y, x, y, 1, 2);
        return;
    }

    scalar_t gx = get_value(grad_x, x, y);
    scalar_t gy = get_value(grad_y, x, y);

    bool is_horizontal = (abs(gy) >= abs(gx)); // horizonal edge

    scalar_t m  = buffer[4];
    scalar_t g1 = is_horizontal ? buffer[1] : buffer[3];
    scalar_t g2 = is_horizontal ? buffer[7] : buffer[5];

    float off = (g1-m)/(g1-g2+0.000000001)-0.5;
    off = fmaxf(fminf(off, 0.5), -0.5);

    if(is_horizontal)
        coord_y += off;
    else
        coord_x += off;

    set_value_channel(result, coord_x, x, y, 0, 2);
    set_value_channel(result, coord_y, x, y, 1, 2);
}

template <typename scalar_t>
static __global__ void resample_kernel(
    scalar_t* result,
    const scalar_t* image,
    const scalar_t* edge,
    float r
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int w = gridDim.x;
    int h = gridDim.y;

    scalar_t off_x = get_value_channel(edge, x, y, 0, 2);
    scalar_t off_y = get_value_channel(edge, x, y, 1, 2);

    float u = (float(x)+0.5 - off_x*r)/float(w);
    float v = (float(y)+0.5 - off_y*r)/float(h);

    pixel<scalar_t> value = sample_pixel(image, u, v);
    set_pixel(result, value, x, y);
}

// C++ API

void mark_edge_op(
    torch::Tensor& result,
    const torch::Tensor& gray,
    const torch::Tensor& grad_x,
    const torch::Tensor& grad_y,
    float thres_min, float thres_max
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = gray.size(0);
    int h = gray.size(2);
    int w = gray.size(3);
    dim3 grid_size(w, h, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gray.scalar_type(), "mark_edge_kernel", [&] {
        mark_edge_kernel<scalar_t><<<grid_size, 9, 9*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            gray.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            thres_min, thres_max
        );
    });
}

void resample_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& edge,
    float r
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(w, h, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "resample_kernel", [&] {
        resample_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            edge.data_ptr<scalar_t>(),
            r
        );
    });
}