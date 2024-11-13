#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void zero_crossing_kernel(
    scalar_t* result,
    const scalar_t* lap,
    float threshold, int mode
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int h = gridDim.x;
    int w = gridDim.y;

    extern __shared__ char __shared_buffer[];
    scalar_t* array = reinterpret_cast<scalar_t*>(__shared_buffer);

    int t = threadIdx.x;
    int im_x = x;
    int im_y = y;

    switch(mode){
    case 0:
        // horizontal and vertical pixel
        if(t != 4){
            if (t/2 == 0)
                im_x += t%2 == 0 ? -1 : 1;
            else
                im_y += t%2 == 0 ? -1 : 1;
        }
        break;
    case 1:
        int i = t / 3;
        int j = t % 3;
        im_x += j-1;
        im_y += i-1;
        break;
    }
    bool out = (im_x < 0 || im_x >= w) || (im_y < 0 || im_y >= h);
    array[t] = out ? 0.0 : get_value(lap, im_x, im_y);
    __syncthreads();

    if(t != 0)
        return;

    #define THRESHOLD(a, b) (fabs(array[a]-array[b]) >= threshold)
    #define ZERO(a, b) (array[a]*array[b]<=0 && THRESHOLD(a, b))

    bool cross;
    switch(mode){
    case 0:
        // horizontal and vertical pixel
        cross = ZERO(1, 0) || ZERO(3, 2);
        break;
    case 1:
        bool e1 = ZERO(1, 7);
        bool e2 = ZERO(3, 5);
        bool e3 = ZERO(0, 8);
        bool e4 = ZERO(2, 6);
        cross = (int(e1)+int(e2)+int(e3)+int(e4)) >= 2;
        break;
    }
    scalar_t edge = cross ? 255.0 : 0;
    set_value(result, edge, x, y);
}

// C++ API

void zero_crossing_op(
    torch::Tensor& result,
    const torch::Tensor& lap,
    float threshold, int mode
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = lap.size(0);
    int h = lap.size(2);
    int w = lap.size(3);
    dim3 grid_size(h, w, b);

    int n = mode == 0 ? 5 : 9;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(lap.scalar_type(), "zero_crossing_kernel", [&] {
        zero_crossing_kernel<scalar_t><<<grid_size, n, n*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            lap.data_ptr<scalar_t>(),
            threshold, mode
        );
    });
}
