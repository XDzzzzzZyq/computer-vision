#include <torch/extension.h>

void uniform_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int size, int pad
);
void gaussian_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float std, int size, int pad
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor uniform_conv(const torch::Tensor& image, int size, int pad) {
    CHECK_INPUT(image);

    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    int e = pad - size;
    torch::Tensor result = torch::empty({b, c, h+2*e, w+2*e}).to(image.device());
    uniform_conv_op(result, image, size, pad);

    return result;
}
torch::Tensor gaussian_conv(const torch::Tensor& image, float std, int size, int pad) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    gaussian_conv_op(result, image, std, size, pad);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uniform_conv", &uniform_conv, "Uniform Convolution");
    m.def("gaussian_conv", &gaussian_conv, "Gaussian Convolution");
}