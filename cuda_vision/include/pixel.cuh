#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
struct pixel{
    scalar_t r;
    scalar_t g;
    scalar_t b;

    __device__ pixel<scalar_t> operator*(float fact){
        return pixel<scalar_t>{scalar_t(r*fact), scalar_t(g*fact), scalar_t(b*fact)};
    }
    __device__ pixel<scalar_t> operator+(const pixel<scalar_t>& tar){
        return pixel<scalar_t>{r+tar.r, g+tar.g, b+tar.b};
    }
    __device__ pixel<scalar_t> operator-(const pixel<scalar_t>& tar){
        return *this + tar * -1.0;
    }
    __device__ pixel<scalar_t> operator+(scalar_t tar){
        return pixel<scalar_t>{r+tar, g+tar, b+tar};
    }
    __device__ pixel<scalar_t> operator-(scalar_t tar){
        return *this + tar * -1.0;
    }
    __device__ __host__ scalar_t grayscale(){
        return 0.299*r + 0.587*g + 0.114*b;
    }
};

template <typename scalar_t>
static __device__ pixel<scalar_t> get_pixel(
    const scalar_t* image,
    int x, int y, int h=0, int w=0
){
    h = h == 0 ? gridDim.x : h;
    w = w == 0 ? gridDim.y : w;
    int chane_off = h * w;
    int batch_off = threadIdx.x * chane_off * 3;
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
    int x, int y, int h=0, int w=0
){
    h = h == 0 ? gridDim.x : h;
    w = w == 0 ? gridDim.y : w;
    int chane_off = h * w;
    int batch_off = threadIdx.x * chane_off * 3;
    int loc = y * h + x;
    image[batch_off + loc] = pixel.r;
    image[batch_off + loc + chane_off] = pixel.g;
    image[batch_off + loc + chane_off*2] = pixel.b;
}

template <typename scalar_t>
static __device__ pixel<scalar_t> get_value(
    const scalar_t* gray,
    int x, int y
){
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    return gray[batch + y * gridDim.x + x];
}