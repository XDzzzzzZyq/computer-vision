#include <torch/extension.h>
#include <iostream>

void to_sat_op(
    torch::Tensor& result,
    const torch::Tensor& image
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor to_sat(const torch::Tensor& image, int order) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image).repeat({1, order, 1, 1});
    to_sat_op(result, image);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_sat", &to_sat, "Construct the Summed Area Table (SAT)");
}