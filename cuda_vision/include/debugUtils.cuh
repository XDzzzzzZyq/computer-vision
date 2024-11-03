#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
void debug_array(const scalar_t* d_array, int size) {
    scalar_t* h_array = new scalar_t[size];

    cudaMemcpy(h_array, d_array, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << "Element " << i << ": " << h_array[i] << std::endl;
    }

    delete[] h_array;
}

void check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}