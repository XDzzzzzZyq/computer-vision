#include <torch/extension.h>
#include <iostream>

void to_sat_op(
    torch::Tensor& result,
    const torch::Tensor& image
);
void get_momentum_op(
    torch::Tensor& result,
    const torch::Tensor& sat,
    int window
);
void get_minmaxmedian_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int window
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

torch::Tensor get_momentum(const torch::Tensor& sat, int window) {
    CHECK_INPUT(sat);

    int B = sat.size(0);
    int O = sat.size(1);
    int H = sat.size(2);
    int W = sat.size(3);
    torch::Tensor result = torch::empty({B, O, H/window, W/window}).to(sat.device());
    get_momentum_op(result, sat, window);

    return result;
}

torch::Tensor get_minmaxmedian(const torch::Tensor& image, int window) {
    CHECK_INPUT(image);

    int B = image.size(0);
    int H = image.size(2);
    int W = image.size(3);
    torch::Tensor result = torch::empty({B, 3, H/window, W/window}).to(image.device());
    get_minmaxmedian_op(result, image, window);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_sat", &to_sat, "Construct the Summed Area Table (SAT)");
    m.def("get_momentum", &get_momentum, "Estimate the centralized momentum up to k-order");
    m.def("get_minmaxmedian", &get_minmaxmedian, "Calculate the Min, Max, and Medium");
}