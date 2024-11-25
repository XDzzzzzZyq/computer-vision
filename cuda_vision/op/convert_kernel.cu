#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void to_grayscale_kernel(
    scalar_t* gray,
    const scalar_t* image
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    pixel<scalar_t> pixel = get_pixel(image, x, y);
    scalar_t grayscale = 0.299*pixel.r + 0.587*pixel.g + 0.114*pixel.b;

    int batch = blockIdx.z * gridDim.x * gridDim.y;
    gray[batch + y * gridDim.x + x] = grayscale;
}

template <typename scalar_t>
static __global__ void invert_kernel(
    scalar_t* result,
    const scalar_t* image,
    bool rgb
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    if(rgb){
        pixel<scalar_t> pixel = get_pixel(image, x, y);
        pixel = (pixel - 255.) * -1.;
        set_pixel(result, pixel, x, y);
    }else{
        scalar_t gray = get_value(image, x, y);
        gray = (gray - 255.) * -1.;
        set_value(result, gray, x, y);
    }
}

template <typename scalar_t>
static __global__ void hash_kernel(
    scalar_t* result,
    const scalar_t* image,
    int seed,
    int prime=982451653
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int mod = pow(10,9) + 7;

    int coor_seed = ((y * gridDim.x + x + seed) * prime) % 255;
    scalar_t gray = get_value(image, x, y);
    gray = fmod((gray + coor_seed) * prime, scalar_t(mod));
    gray/= mod/255.0;
    set_value(result, gray, x, y);
}

template <typename scalar_t>
static __global__ void threshold_kernel(
    scalar_t* result,
    const scalar_t* image,
    float threshold, bool rgb
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    if(rgb){
        pixel<scalar_t> pixel = get_pixel(image, x, y);
        pixel.r = pixel.r < threshold ? 0.0 : 255.0;
        pixel.g = pixel.g < threshold ? 0.0 : 255.0;
        pixel.b = pixel.b < threshold ? 0.0 : 255.0;
        set_pixel(result, pixel, x, y);
    }else{
        scalar_t gray = get_value(image, x, y);
        gray = gray < threshold ? 0.0 : 255.0;
        set_value(result, gray, x, y);
    }
}

template <typename scalar_t>
static __global__ void random_threshold_kernel(
    scalar_t* result,
    const scalar_t* image,
    const scalar_t* noise
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    scalar_t gray = get_value(image, x, y);
    scalar_t rand = get_value(noise, x, y);
    gray = rand < gray ? 255.0 : 0.0;
    set_value(result, gray, x, y);
}

template <typename scalar_t>
static __global__ void matrix_dither_kernel(
    scalar_t* result,
    const scalar_t* image,
    const scalar_t* index,
    int n
) {
    int h = gridDim.x * n;
    int w = gridDim.y * n;
    int t = threadIdx.x;
    int bx = t % n;
    int by = t / n;
    int x = blockIdx.x * n + bx;
    int y = blockIdx.y * n + by;

    scalar_t gray = get_value(image, x, y, w, h);
    scalar_t indx = index[by * n + bx];
    scalar_t thrs = (indx+0.5) / (n*n) * 255.0;
    gray = gray > thrs ? 255.0 : 0.0;
    set_value(result, gray, x, y, w, h);
}

template <typename scalar_t>
static __global__ void error_diffusion_kernel(
    scalar_t* result,
    scalar_t* error,
    const scalar_t* image,
    const scalar_t* diffuse,
    float threshold, bool serpentine,
    int n, int h, int w
) {
    int t = threadIdx.x;
    int off_idx = 1+n*n/2;
    int off_x = (t+off_idx) % n - 1;
    int off_y = (t+off_idx) / n - 1;

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            int inv_x = serpentine ? y%2==0?x:w-x-1 : x;
            scalar_t f = get_value(image, inv_x, y, w, h);
            scalar_t e = get_value(error, inv_x, y, w, h);
            scalar_t b = (f+e) < threshold ? 0.0 : 255.0;
            e += f - b;

            int loc_x = x + off_x;
            int loc_y = y + off_y;
            if(serpentine)
                loc_x = y%2==0?loc_x:w-loc_x-1;
            if(!IS_OUT(loc_x, loc_y, w, h)){
                scalar_t weight = diffuse[t+off_idx];
                set_value(error, e * weight, loc_x, loc_y, w, h);
            }
            __syncthreads();
            if(t == 0)
                set_value(result, b, inv_x, y, w, h);
        }
    }
}

// C++ API

void to_grayscale_op(
    torch::Tensor& gray,
    const torch::Tensor& image
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "to_grayscale_kernel", [&] {
        to_grayscale_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            gray.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>()
        );
    });
}

void invert_op(
    torch::Tensor& result,
    const torch::Tensor& image
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "invert_kernel", [&] {
        invert_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            c == 3
        );
    });
}

void hash_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int seed
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "hash_kernel", [&] {
        hash_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            seed
        );
    });
}

void threshold_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float threshold
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "threshold_kernel", [&] {
        threshold_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            threshold, c == 3
        );
    });
}

void random_threshold_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& noise
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "random_threshold_kernel", [&] {
        random_threshold_kernel<scalar_t><<<grid_size, 1, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            noise.data_ptr<scalar_t>()
        );
    });
}

void matrix_dither_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& index
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    int n = index.size(0);
    dim3 grid_size(h/n, w/n, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "matrix_dither_kernel", [&] {
        matrix_dither_kernel<scalar_t><<<grid_size, n*n, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            index.data_ptr<scalar_t>(),
            n
        );
    });
}

void error_diffusion_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& diffuse,
    float threshold, bool serpentine
){
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    int n = diffuse.size(0);
    dim3 grid_size(1, 1, b);

    torch::Tensor error = torch::zeros_like(image);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "error_diffusion_kernel", [&] {
        error_diffusion_kernel<scalar_t><<<grid_size, n*n/2, 0, stream>>>(
            result.data_ptr<scalar_t>(),
            error.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            diffuse.data_ptr<scalar_t>(),
            threshold, serpentine,
            n, h, w
        );
    });
}