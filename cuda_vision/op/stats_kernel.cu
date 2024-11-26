#include <torch/types.h>
#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void fill_momentum_kernel(
    scalar_t* result,
    const scalar_t* image
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int o = threadIdx.x;
    int c = blockDim.x;

    scalar_t value = get_value(image, x, y);
    scalar_t momen = pow(value, scalar_t(o+1));
    set_value_channel(result, momen, x, y, o, c);
}

template <typename scalar_t>
static __global__ void semi_sat_kernel(
    scalar_t* result,
    const scalar_t* image
) {
    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int i = threadIdx.x;
    int y = blockIdx.x;
    int x0= i*2;
    int x1= i*2+1;
    int o = blockIdx.y;
    int c = gridDim.y;
    int w = blockDim.x*2;
    int h = gridDim.x;
    
    unsigned int rd_id;
	unsigned int wr_id;
	unsigned int mask;
	buffer[x0] = get_value_channel(image, x0, y, o, c, w, h);
	buffer[x1] = get_value_channel(image, x1, y, o, c, w, h);
	__syncthreads();

	int steps = int(log2(double(w))) + 1;
	for (int step = 0; step < steps; step++)
	{
		mask = (1 << step) - 1;
		rd_id = ((i >> step) << (step + 1)) + mask;
		wr_id = rd_id + 1 + (i & mask);
		buffer[wr_id] += buffer[rd_id];
		__syncthreads();
	}

	set_value_channel(result, buffer[x0], y, x0, o, c, h, w);  // transpose
	set_value_channel(result, buffer[x1], y, x1, o, c, h, w);
}

// C++ API

void to_sat_op(
    torch::Tensor& result,
    const torch::Tensor& image
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = image.size(0);
    int o = result.size(1);
    int h = image.size(2);
    int w = image.size(3);
    dim3 grid_size0(h, w, b);
    dim3 grid_size1(h, o, b);

    torch::Tensor temp1 = torch::empty_like(result);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "fill_momentum_kernel", [&] {
        fill_momentum_kernel<scalar_t><<<grid_size0, o, 0, stream>>>(
            temp1.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>()
        );
    });

    torch::Tensor temp2 = torch::empty_like(result);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_sat_kernel", [&] {
        semi_sat_kernel<scalar_t><<<grid_size1, w/2, w*sizeof(scalar_t), stream>>>(
            temp2.data_ptr<scalar_t>(),
            temp1.data_ptr<scalar_t>()
        );
    });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "semi_sat_kernel", [&] {
        semi_sat_kernel<scalar_t><<<grid_size1, w/2, w*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            temp2.data_ptr<scalar_t>()
        );
    });
}