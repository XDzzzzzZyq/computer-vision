#include <torch/types.h>
#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pixelUtils.cuh"
#include "arrayUtils.cuh"

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

template <typename scalar_t>
static __global__ void estimate_momentum_kernel(
    scalar_t* result,
    const scalar_t* sat,
    int window, int sat_w, int sat_h
) {
    extern __shared__ char __shared_buffer[];
    scalar_t* buffer = reinterpret_cast<scalar_t*>(__shared_buffer);

    int x = blockIdx.x;
    int y = blockIdx.y;
    int o = threadIdx.x;
    int c = blockDim.x;
    int n = window * window;

    int x0 = -1+x*window;
    int x1 = -1+(x+1)*window;
    int y0 = -1+y*window;
    int y1 = -1+(y+1)*window;

#define _GET_SAT(_x, _y) IS_OUT(_x, _y, sat_w, sat_h) ? 0.0 : get_value_channel(sat, _x, _y, o, c, sat_w, sat_h)

    scalar_t top_left = _GET_SAT(x0, y0);
    scalar_t top_right= _GET_SAT(x1, y0);
    scalar_t dwn_left = _GET_SAT(x0, y1);
    scalar_t dwn_right= _GET_SAT(x1, y1);

    buffer[o] = (dwn_right - dwn_left - top_right + top_left)/scalar_t(n);
    __syncthreads();

    scalar_t momentum = 0.0;
    switch(o+1){
    case 1: // E[X]
        momentum = buffer[o];
        break;
    case 2: // V[X] = E[X^2] - E[X]^2
        momentum = buffer[o] - buffer[o-1]*buffer[o-1];
        break;
    case 3:
        momentum = buffer[o] - 3*buffer[o-2]*buffer[o-1] + 2*pow(buffer[o-2], scalar_t(3.0));
        momentum/= pow(buffer[o-1]-buffer[o-2]*buffer[o-2], scalar_t(1.5));
        break;
    default:
        momentum = buffer[o];
        break;
    }

    set_value_channel(result, momentum, x, y, o, c);
}

template <typename scalar_t>
static __global__ void minmaxmedian_kernel(
    scalar_t* result,
    const scalar_t* gray,
    int window, int w, int h
) {
    extern __shared__ char __shared_buffer[];
    scalar_t* array = reinterpret_cast<scalar_t*>(__shared_buffer);

    int x = blockIdx.x;
    int y = blockIdx.y;
    int n = window * window;

    int t = threadIdx.x;
    int im_x = x*window+t%window;
    int im_y = y*window+t/window;

    array[t] = get_value(gray, im_x, im_y, w, h);
    __syncthreads();

    if(t == 0)
        arr::quicksort(array, 0, n-1);
    __syncthreads();

    switch(t){
    case 0:
        set_value_channel(result, array[0], x, y, t, 3);
        break;
    case 1:
        set_value_channel(result, array[n-1], x, y, t, 3);
        break;
    case 2:
        if (n%2 == 0)
            set_value_channel(result, (array[n/2-1]+array[n/2])/2, x, y, t, 3);
        else
            set_value_channel(result, array[n/2], x, y, t, 3);
        break;
    }
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

void get_momentum_op(
    torch::Tensor& result,
    const torch::Tensor& sat,
    int window
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = result.size(0);
    int o = result.size(1);
    int h = sat.size(2);
    int w = sat.size(3);
    dim3 grid_size0(h/window, w/window, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sat.scalar_type(), "estimate_momentum_kernel", [&] {
        estimate_momentum_kernel<scalar_t><<<grid_size0, o, o*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            sat.data_ptr<scalar_t>(),
            window, w, h
        );
    });
}

void get_minmaxmedian_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int window
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = result.size(0);
    int h = image.size(2);
    int w = image.size(3);
    int n = window * window;
    dim3 grid_size0(h/window, w/window, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "minmaxmedian_kernel", [&] {
        minmaxmedian_kernel<scalar_t><<<grid_size0, n, n*sizeof(scalar_t), stream>>>(
            result.data_ptr<scalar_t>(),
            image.data_ptr<scalar_t>(),
            window, w, h
        );
    });
}
