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

    int* ptr;
    cudaMalloc(&ptr, size);
    int block_size = 256;
    int grid_size = num / block_size;
    init_array<int><<<grid_size, block_size, 0, stream>>>(
        ptr, num, def
    );

    return ptr;
}