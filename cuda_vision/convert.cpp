#include <torch/extension.h>
#include <vector>

void to_grayscale_op(
    torch::Tensor& gray,
    const torch::Tensor& image
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor to_grayscale(const torch::Tensor& image) {
    CHECK_CONTIGUOUS(image);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    torch::Tensor gray = torch::empty({b, 1, h, w}).to(image.device());
    to_grayscale_op(gray, image);

    return gray;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_grayscale", &to_grayscale, "Convert RGB image to grayscale");
}