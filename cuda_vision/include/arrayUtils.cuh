#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void init_array(scalar_t* array, int num, scalar_t value) {
    int loc = blockIdx.x * blockDim.x + threadIdx.x;
    if (loc <= num)
        array[loc] = value;
}

template <typename scalar_t>
scalar_t* make_array(int num, scalar_t def=0){
    int size = num * sizeof(scalar_t);
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    scalar_t* ptr;
    cudaMalloc(&ptr, size);
    int block_size = 256;
    int grid_size = num / block_size + 1;
    init_array<scalar_t><<<grid_size, block_size, 0, stream>>>(
        ptr, num, def
    );

    return ptr;
}

namespace arr{
    template <typename scalar_t>
    __host__ __device__ scalar_t min(scalar_t* array, int num, int begin=0) {
        scalar_t min_v = array[0];
        for(int i = begin; i < num; i++)
            min_v = ::min(min_v, array[i]);
        return min_v;
    }

    template <typename scalar_t>
    __host__ __device__ scalar_t max(scalar_t* array, int num, int begin=0) {
        scalar_t max_v = array[0];
        for(int i = begin; i < num; i++)
            max_v = ::max(max_v, array[i]);
        return max_v;
    }

    template <typename scalar_t>
    __host__ __device__ scalar_t minmax(scalar_t* array, scalar_t* temp, int num) {
        int m = (num+1)/2;
        for(int i = 0; i < num-m+1; i++)
            temp[i] = arr::max(array, m+i, i);
        return arr::min(temp, num-m+1);
    }

    template <typename scalar_t>
    __host__ __device__ scalar_t maxmin(scalar_t* array, scalar_t* temp, int num) {
        int m = (num+1)/2;
        for(int i = 0; i < num-m+1; i++)
            temp[i] = arr::min(array, m+i, i);
        return arr::max(temp, num-m+1);
    }

    template <typename scalar_t>
    __host__ __device__ void sort(scalar_t* array, int num) {

    }

    template <typename scalar_t>
    __host__ __device__ scalar_t median(scalar_t* array, int num) {
        arr::sort(array, num);
        return array[(num-1)/2];
    }
};

template <typename scalar_t>
void debug_array(const scalar_t* d_array, int size) {
    scalar_t* h_array = new scalar_t[size];

    cudaMemcpy(h_array, d_array, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << "Element " << i << ": " << h_array[i] << std::endl;
    }

    delete[] h_array;
}