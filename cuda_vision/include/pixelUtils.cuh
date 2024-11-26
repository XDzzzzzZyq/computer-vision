#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#define IS_OUT1D(x, w) (x < 0 || x >= w)
#define IS_OUT(x, y, w, h) (IS_OUT1D(x, w) || IS_OUT1D(y, h))

template <typename scalar_t>
struct pixel{
    scalar_t r=0;
    scalar_t g=0;
    scalar_t b=0;

    __host__ __device__ pixel<scalar_t> operator+(scalar_t tar) const {
        return pixel<scalar_t>{r + tar, g + tar, b + tar};
    }
    __host__ __device__ pixel<scalar_t> operator-(scalar_t tar) const {
        return pixel<scalar_t>{r - tar, g - tar, b - tar};
    }
    __host__ __device__ pixel<scalar_t> operator*(scalar_t fact) const {
        return pixel<scalar_t>{r * fact, g * fact, b * fact};
    }
    __host__ __device__ pixel<scalar_t> operator/(scalar_t fact) const {
        return pixel<scalar_t>{r / fact, g / fact, b / fact};
    }
    __host__ __device__ pixel<scalar_t> operator+(const pixel<scalar_t>& tar) const {
        return pixel<scalar_t>{r + tar.r, g + tar.g, b + tar.b};
    }
    __host__ __device__ pixel<scalar_t> operator-(const pixel<scalar_t>& tar) const {
        return pixel<scalar_t>{r - tar.r, g - tar.g, b - tar.b};
    }
    __host__ __device__ pixel<scalar_t> operator*(const pixel<scalar_t>& tar) const {
        return pixel<scalar_t>{r * tar.r, g * tar.g, b * tar.b};
    }
    __host__ __device__ pixel<scalar_t> operator/(const pixel<scalar_t>& tar) const {
        return pixel<scalar_t>{r / tar.r, g / tar.g, b / tar.b};
    }
    __device__ __host__ scalar_t grayscale(){
        return 0.299*r + 0.587*g + 0.114*b;
    }
};

template <typename scalar_t>
static __device__ pixel<scalar_t> get_pixel(
    const scalar_t* image,
    int x, int y, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int chane_off = h * w;
    int batch_off = blockIdx.z * chane_off * 3;
    int loc = y * h + x;
    scalar_t r = image[batch_off + loc];
    scalar_t g = image[batch_off + loc + chane_off];
    scalar_t b = image[batch_off + loc + chane_off*2];
    return pixel<scalar_t>{r, g, b};
}

template <typename scalar_t>
static __device__ void set_pixel(
    scalar_t* image,
    pixel<scalar_t> pixel,
    int x, int y, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int chane_off = h * w;
    int batch_off = blockIdx.z * chane_off * 3;
    int loc = y * h + x;
    image[batch_off + loc] = pixel.r;
    image[batch_off + loc + chane_off] = pixel.g;
    image[batch_off + loc + chane_off*2] = pixel.b;
}

template <typename scalar_t>
static __device__ scalar_t get_value(
    const scalar_t* gray,
    int x, int y, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int batch_off = blockIdx.z * h * w;
    return gray[batch_off + y * h + x];
}

template <typename scalar_t>
static __device__ void set_value(
    scalar_t* gray,
    scalar_t value,
    int x, int y, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int batch_off = blockIdx.z * h * w;
    gray[batch_off + y * h + x] = value;
}

template <typename scalar_t>
static __device__ scalar_t get_value_channel(
    const scalar_t* gray,
    int x, int y, int c, int num_ch, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;

    int channel_off = h * w;
    int batch_off = channel_off * num_ch * blockIdx.z;
    return gray[batch_off + c * channel_off + y * h + x];
}

template <typename scalar_t>
static __device__ void set_value_channel(
    scalar_t* gray,
    scalar_t value,
    int x, int y, int c, int num_ch, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int channel_off = h * w;
    int batch_off = channel_off * num_ch * blockIdx.z;
    gray[batch_off + c * channel_off + y * h + x] = value;
}

template <typename scalar_t>
static __device__ scalar_t sample_value(
    const scalar_t* gray,
    float u, float v, int w=0, int h=0
){
    w = w == 0 ? gridDim.y : w;
    h = h == 0 ? gridDim.x : h;
    int batch_off = blockIdx.z * h * w;

    int x0 = int(u*w-0.5);
    int y0 = int(v*h-0.5);
    int x1 = x0+1;
    int y1 = y0+1;

#define _GET(x, y) (IS_OUT(x, y, w, h) ? 0.0 : gray[batch_off + y * h + x])
    scalar_t f00 = _GET(x0, y0);
    scalar_t f10 = _GET(x1, y0);
    scalar_t f01 = _GET(x0, y1);
    scalar_t f11 = _GET(x1, y1);

    float fx = u*w-0.5-x0;
    float fy = v*h-0.5-y0;

#define LINEAR(a, b, f) (f*b + (1.0-f)*a)
    scalar_t f0 = LINEAR(f00, f10, fx);
    scalar_t f1 = LINEAR(f01, f11, fx);

    return LINEAR(f0, f1, fy);
}