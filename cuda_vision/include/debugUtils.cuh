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

void check_max_grid_size() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        std::cout << "Device " << device << ": " << device_prop.name << "\n";
        std::cout << "  Max grid size: ("
                  << device_prop.maxGridSize[0] << ", "
                  << device_prop.maxGridSize[1] << ", "
                  << device_prop.maxGridSize[2] << ")\n";
    }
}

void check_max_block_size() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        std::cout << "Device " << device << ": " << device_prop.name << "\n";
        std::cout << "  Max threads per block: " << device_prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per block dimension: ("
                  << device_prop.maxThreadsDim[0] << ", "
                  << device_prop.maxThreadsDim[1] << ", "
                  << device_prop.maxThreadsDim[2] << ")\n";
    }
}
