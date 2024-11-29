#include <torch/extension.h>
#include <iostream>

void simple_transform_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float arg1, float arg2,
    int mode
);
void custom_transform_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& matrix
);
void disk_warp_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    bool inverse
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor simple_transform(const torch::Tensor& image, float arg1, float arg2, int mode, int w_new, int h_new) {
    CHECK_INPUT(image);

    int B = image.size(0);
    torch::Tensor result = torch::empty({B, 1, h_new, w_new}).to(image.device());
    simple_transform_op(result, image, arg1, arg2, mode);

    return result;
}

torch::Tensor custom_transform(const torch::Tensor& image, const torch::Tensor& matrix, int w_new, int h_new) {
    CHECK_INPUT(image);
    CHECK_INPUT(matrix);

    int B = image.size(0);
    torch::Tensor result = torch::empty({B, 1, h_new, w_new}).to(image.device());
    custom_transform_op(result, image, matrix);

    return result;
}

torch::Tensor disk_warp(const torch::Tensor& image, bool inverse) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    disk_warp_op(result, image, inverse);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simple_transform", &simple_transform, "Translate the images by simple operation");
    m.def("custom_transform", &custom_transform, "Transform the images by given matrix");
    m.def("disk_warp", &disk_warp, "Forward and Backward Disk Warping");
}