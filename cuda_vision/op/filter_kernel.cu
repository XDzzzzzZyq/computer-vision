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
#include "debugUtils.cuh"

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
    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int im_x = x-pad+t;
    int im_y = y+size-pad;
    scalar_t p = IS_OUT1D(im_x, w) ? 0 : get_value(gray, im_x, im_y, w, h);

    float weight = semi_kernel[t];
    buffer[t] = p * weight;
    __syncthreads();

    if(t != 0)
        return;

    scalar_t r = 0.0;
    for(int i = 0; i<2*size+1; i++){
        r += buffer[i];
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
    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int im_x = x-pad+t;
    int im_y = y+size-pad;
    scalar_t center = get_value(gray, x+e, y+e, w, h);
    scalar_t p = IS_OUT1D(im_x, w) ? 0 : get_value(gray, im_x, im_y, w, h);

    float weight1 = semi_kernel[t];
    float weight2 = gaus(center-p, std);
    buffer[t]            = p * weight1 * weight2;
    buffer[t + 2*size+1] = weight1 * weight2;
    __syncthreads();

    if(t != 0)
        return;

    scalar_t r = 0.0;
    scalar_t norm = 0.0;
    for(int i = 0; i<2*size+1; i++){
        r += buffer[i];
        norm += buffer[i + 2*size+1];
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
static __global__ void full_conv_gray_kernel(
    scalar_t* result,
    const scalar_t* gray,
    const float* kernel,
    int size, int pad
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x;
    int w = gridDim.y - 2 * (pad - size);
    int n = size*2+1;
    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int i = t / n;
    int j = t - i * n;

    int im_x = x-pad+i;
    int im_y = y-pad+j;
    scalar_t p = IS_OUT1D(im_x, w) ? 0 : get_value(gray, im_x, im_y, w, h);

    float weight = kernel[t];
    buffer[t] = p * weight;
    __syncthreads();

    if(t != 0)
        return;

    scalar_t r = 0.0;
    for(int i = 0; i<(2*size+1)*(2*size+1); i++){
        r += buffer[i];
    }
    set_value(result, r, x, y);
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

    extern __shared__ char __shared_buffer[];
    scalar_t* array = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int i = t / n;
    int j = t - i * n;

    int im_x = x-pad+i;
    int im_y = y-pad+j;

    array[t] = IS_OUT(im_x, im_y, w, h) ? 0.0 : get_value(gray, im_x, im_y, w, h);
    __syncthreads();

    if(t != 0)
        return;

    scalar_t median = 0.0;
    if(pseudo){
        scalar_t* temp  = (scalar_t*)malloc((len-m+1) * sizeof(scalar_t));
        scalar_t minmax_v = arr::minmax((scalar_t*)array, temp, len);
        scalar_t maxmin_v = arr::maxmin((scalar_t*)array, temp, len);
        median = (minmax_v + maxmin_v)/2.0;
        free(temp);
    }else{
        median = arr::median((scalar_t*)array, len);
    }

    set_value(result, median, x, y);
}

template <typename scalar_t>
static __global__ void pattern_match_kernel(
    scalar_t* mark,
    const scalar_t* image,
    const scalar_t* patterns,
    bool conditional
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x;
    int w = gridDim.y;
    int n = blockDim.x;

    extern __shared__ char __shared_buffer[];
    scalar_t storage[9];
    for(int i = 0; i<3; i++){
        for(int j = 0; j<3; j++){
            int im_x = x+j-1;
            int im_y = y+i-1;
            storage[i*3+j] = IS_OUT(im_x, im_y, w, h) ? (conditional ? 255.0 : 0.0) : get_value(image, im_x, im_y);
        }
    }

    if (storage[4] == 0.0){
        set_value(mark, scalar_t(0.0), x, y);
        return;
    }

    int t = threadIdx.x;
    bool match = true;
    if (conditional){
        for(int i = 0; i<9; i++){
            match &= patterns[t*9+i]*255.0 == storage[i];
        }
    }else{
        bool use_abc = false;
        int abc_count = 0;
        for(int i = 0; i<9; i++){
            int p = int(patterns[t*9+i]);
            switch(p){
            case 1:
            case 0:
                match &= p*255.0 == storage[i];
                break;
            case -1:
                continue;
            case -2:
                use_abc = true;
                abc_count += storage[i]==255. ? 1 : 0;
            }
        }
        match &= !use_abc || (use_abc && abc_count>0);
    }

    __shared_buffer[t] = char(match);
    __syncthreads();

    if(t != 0)
        return;

    match = false;
    for(int i = 0; i<n; i++){
        match |= __shared_buffer[i] == 1;
    }
    scalar_t v = match ? 255.0 : 0;
    set_value(mark, v, x, y);
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
    int l = 2*size+1;

    torch::Tensor temp = torch::empty({b, c, w+2*e, h}).to(image.device()); // transpose
    dim3 grid_size1(h,     w+2*e, b);
    dim3 grid_size2(h+2*e, w+2*e, b);

    if(c == 3){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_kernel", [&] {
            semi_conv_kernel<scalar_t><<<grid_size1, l, l*sizeof(scalar_t), stream>>>(
                temp.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_kernel", [&] {
            semi_conv_kernel<scalar_t><<<grid_size2, l, l*sizeof(scalar_t), stream>>>(
                result.data_ptr<scalar_t>(),
                temp.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
    }else if (c == 1){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_gray_kernel", [&] {
            semi_conv_gray_kernel<scalar_t><<<grid_size1, l, l*sizeof(scalar_t), stream>>>(
                temp.data_ptr<scalar_t>(),
                image.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_conv_gray_kernel", [&] {
            semi_conv_gray_kernel<scalar_t><<<grid_size2, l, l*sizeof(scalar_t), stream>>>(
                result.data_ptr<scalar_t>(),
                temp.data_ptr<scalar_t>(),
                kernel,
                size, pad
            );
        });
    }
}

void full_conv_op(
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
    int l = (2*size+1)*(2*size+1);
    int e = pad - size;
    dim3 grid_size(h+2*e, w+2*e, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "full_conv_gray_kernel", [&] {
        full_conv_gray_kernel<scalar_t><<<grid_size, l, l*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            kernel,
            size, pad
        );
    });
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

void custom_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& kernel, int pad
) {
    int ndim = kernel.dim();
    int size = kernel.size(0) / 2;
    switch(ndim){
    case 1:
        separable_conv_op(result, image, kernel.data_ptr<float>(), size, pad);
        break;
    case 2:
        full_conv_op(result, image, kernel.data_ptr<float>(), size, pad);
        break;
    }
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
    int n = 2*size+1;

    cudaDeviceSetLimit(cudaLimitStackSize, n*n*1024+1024);

    dim3 grid_size(h+2*e, w+2*e, b);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "median_kernel", [&] {
        median_kernel<scalar_t><<<grid_size, n*n, n*n*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            size, pad, pseudo
        );
    });
}

#define PI 3.14159265358979323846
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
    int l = 2*size+1;

    float* kernel = make_array<float>(2*size+1, 0);
    get_gaussian_kernel<<<1, 2*size+1, (2*size+1)*sizeof(float), stream>>>(kernel, std_k, size);

    torch::Tensor temp = torch::empty({b, c, w+2*e, h}).to(image.device()); // transpose
    dim3 grid_size1(h,     w+2*e, b);
    dim3 grid_size2(h+2*e, w+2*e, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_bilateral_conv_gray_kernel", [&] {
        semi_bilateral_conv_gray_kernel<scalar_t><<<grid_size1, l, 2*l*sizeof(scalar_t), stream>>>(
            temp.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            kernel, std_i,
            size, pad
        );
    });
    float gau_2 = 0.0;
    for(int i = -size; i<size+1; i++){
        float gau = gaus(i, std_k);
        gau_2 += gau * gau;
    }
    float fact = gau_2 / (2*PI*std_i*std_i + 4*PI*std_k*std_k);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_bilateral_conv_gray_kernel", [&] {
        semi_bilateral_conv_gray_kernel<scalar_t><<<grid_size2, l, 2*l*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            temp.data_ptr<scalar_t>(),
            kernel, std_i,// / sqrtf(fact),
            size, pad
        );
    });
}

void pattern_match_op(
    torch::Tensor& mark,
    const torch::Tensor& image,
    const torch::Tensor& pattern,
    bool cond
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    int n = pattern.size(0);

    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "pattern_match_kernel", [&] {
        pattern_match_kernel<scalar_t><<<grid_size, n, n*sizeof(char), stream>>>(
            mark.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            pattern.data_ptr<scalar_t>(),
            cond
        );
    });
}