#include <torch/extension.h>
#include <iostream>

void minmax_scale_op(
    torch::Tensor& result,
    const torch::Tensor& image
);
void histo_equal_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int k
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor minmax_scale(const torch::Tensor& image) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    minmax_scale_op(result, image);

    return result;
}

torch::Tensor histo_equal(const torch::Tensor& image, int k) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    histo_equal_op(result, image, k);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("minmax_scale", &minmax_scale, "Full Range Linear Scaling Method");
    m.def("uniform_equalize", &histo_equal, "Uniform Histogram Equalization Method");
}