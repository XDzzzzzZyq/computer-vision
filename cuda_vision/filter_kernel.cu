#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include "pixelUtils.cuh"
#include "arrayUtils.cuh"

__host__ __device__ float gaus(float x, float std){
    return expf(- (x * x) / (2.0f * std * std)) / (sqrtf(2.0f * M_PI) * std);
}

static __global__ void get_gaussian_kernel(
    float* kernel,
    float std, int size
) {
    int idx = threadIdx.x;
    int x = idx - size;
    extern __shared__ float sharedKernel[];

    float value = gaus(x, std);
    sharedKernel[idx] = value;
    __syncthreads();

    float sum = 0;
    for(int i = 0; i < blockDim.x; i++)
        sum += sharedKernel[i];

    kernel[idx] = value / sum;
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
static __global__ void semi_bilateral_conv_gray_kernel(
    scalar_t* result,
    const scalar_t* gray,
    const float* semi_kernel,
    float std, int size, int pad
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x;
    int e = size-pad;
    int w = gridDim.y + 2 * e;

    scalar_t center = get_value(gray, x+e, y+e, h, w);
    scalar_t r = 0.0;
    scalar_t norm = 0.0;
    for(int i = -size; i<size+1; i++){
        int im_x = x+e+i;
        int im_y = y+e;

        scalar_t p = (im_x < 0 || im_x >= w) ? 0 : get_value(gray, im_x, im_y, h, w);
        float weight1 = semi_kernel[i+size];
        float weight2 = gaus(center-p, std);
        r += p * weight1 * weight2;
        norm += weight1 * weight2;
    }
    r /= norm;
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
    // TODO: to be finished.
    get_pixel(image, x, y);
}

template <typename scalar_t>
static __global__ void median_kernel(
    scalar_t* result,
    const scalar_t* gray,
    int size, int pad, bool pseudo
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x - 2 * (pad - size);
    int w = gridDim.y - 2 * (pad - size);
    int n = size*2+1;
    int len = n*n;
    int m = (len+1)/2;

    scalar_t* array = (scalar_t*)malloc(len * sizeof(scalar_t));
    for(int i = -size; i<size+1; i++){
        for(int j = -size; j<size+1; j++){
            int im_x = x+size-pad+i;
            int im_y = y+size-pad+j;

            bool out = (im_x < 0 || im_x >= w) || (im_y < 0 || im_y >= h);
            array[(i+size)*n + j+size] = out ? 0.0 : get_value(gray, im_x, im_y, h, w);
        }
    }

    scalar_t median = 0.0;
    if(pseudo){
        scalar_t* temp  = (scalar_t*)malloc((len-m+1) * sizeof(scalar_t));
        scalar_t minmax_v = arr::minmax(array, temp, len);
        scalar_t maxmin_v = arr::maxmin(array, temp, len);
        median = (minmax_v + maxmin_v)/2.0;
        free(temp);
    }else{
        median = arr::median(array, len);
    }

    set_value(result, median, x, y);
    free(array);
}

// C++ API

void separable_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const float* kernel,
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

void uniform_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int size, int pad
) {
    float* kernel = make_array<float>(2*size+1, 1.0/(2.0*size+1.0));
    separable_conv_op(result, image, kernel, size, pad);
    cudaFree(kernel);
}

void gaussian_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float std, int size, int pad
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    float* kernel = make_array<float>(2*size+1, 0);
    get_gaussian_kernel<<<1, 2*size+1, (2*size+1)*sizeof(float), stream>>>(kernel, std, size);
    separable_conv_op(result, image, kernel, size, pad);
    cudaFree(kernel);
}

void median_filter_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int size, int pad, bool pseudo
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    int e = pad - size;

    dim3 grid_size(h+2*e, w+2*e, 1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "median_kernel", [&] {
        median_kernel<scalar_t><<<grid_size, b, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            size, pad, pseudo
        );
    });
}

void bilateral_filter_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float std_k, float std_i, int size, int pad
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    int e = pad - size;

    float* kernel = make_array<float>(2*size+1, 0);
    get_gaussian_kernel<<<1, 2*size+1, (2*size+1)*sizeof(float), stream>>>(kernel, std_k, size);

    torch::Tensor temp = torch::empty({b, c, w+2*e, h}).to(image.device()); // transpose
    dim3 grid_size1(h,     w+2*e, 1);
    dim3 grid_size2(h+2*e, w+2*e, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_bilateral_conv_gray_kernel", [&] {
        semi_bilateral_conv_gray_kernel<scalar_t><<<grid_size1, b, 0, stream>>>(
            temp.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            kernel, std_i,
            size, pad
        );
    });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_bilateral_conv_gray_kernel", [&] {
        semi_bilateral_conv_gray_kernel<scalar_t><<<grid_size2, b, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            temp.data_ptr<scalar_t>(),
            kernel, std_i,
            size, pad
        );
    });
}