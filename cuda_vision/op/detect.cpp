#include <torch/extension.h>

void zero_crossing_op(
    torch::Tensor& result,
    const torch::Tensor& lap,
    float threshold, int mode
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor zero_crossing(const torch::Tensor& lap, float threshold, int mode) {
    CHECK_INPUT(lap);

    torch::Tensor result = torch::empty_like(lap);
    zero_crossing_op(result, lap, threshold, mode);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("zero_crossing", &zero_crossing, "Uniform Convolution");
}