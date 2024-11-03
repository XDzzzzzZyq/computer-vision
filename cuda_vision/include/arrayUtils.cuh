#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

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
    __host__ __device__ void swap(scalar_t* data, int i, int j) {
        scalar_t temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    template <typename scalar_t>
    __host__ __device__ int partition(scalar_t* data, int left, int right) {
        scalar_t pivot = data[right];
        int i = left - 1;

        for (int j = left; j < right; j++) {
            if (data[j] <= pivot) {
                i++;
                arr::swap(data, i, j);
            }
        }

        arr::swap(data, i + 1, right);
        return i + 1;
    }

    template <typename scalar_t>
    __host__ __device__ void quicksort(scalar_t* data, int left, int right) {
        if (left < right) {
            scalar_t pivot = arr::partition(data, left, right);
            arr::quicksort(data, left, pivot - 1);
            arr::quicksort(data, pivot + 1, right);
        }
    }

    template <typename scalar_t>
    __host__ __device__ scalar_t median(scalar_t* array, int num) {
        arr::quicksort(array, 0, num-1);
        return array[(num-1)/2];
    }
};

