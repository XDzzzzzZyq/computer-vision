#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void init_array(scalar_t* array, scalar_t value) {
    array[threadIdx.x] = value;
}

template <typename scalar_t>
scalar_t* make_array(int num, scalar_t def=0){
    int size = num * sizeof(scalar_t);
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int* ptr;
    cudaMalloc(&ptr, size);
    init_array<int><<<1, num, 0, stream>>>(
        ptr, def
    );

    return ptr;
}