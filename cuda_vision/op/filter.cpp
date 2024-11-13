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
void median_filter_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int size, int pad, bool pseudo
);
void bilateral_filter_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float std_k, float std_i, int size, int pad
);
void custom_conv_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& kernel,int pad
);
void conditional_match_op(
    torch::Tensor& mark,
    const torch::Tensor& image,
    const torch::Tensor& pattern
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor get_conv_empty(const torch::Tensor& image, int size, int pad) {
    int b = image.size(0);
    int c = image.size(1);
    int h = image.size(2);
    int w = image.size(3);
    int e = pad - size;

    return torch::empty({b, c, h+2*e, w+2*e}).to(image.device());
}

torch::Tensor uniform_conv(const torch::Tensor& image, int size, int pad) {
    CHECK_INPUT(image);

    torch::Tensor result = get_conv_empty(image, size, pad);
    uniform_conv_op(result, image, size, pad);

    return result;
}
torch::Tensor gaussian_conv(const torch::Tensor& image, float std, int size, int pad) {
    CHECK_INPUT(image);

    torch::Tensor result = get_conv_empty(image, size, pad);
    gaussian_conv_op(result, image, std, size, pad);

    return result;
}
torch::Tensor median_filter(const torch::Tensor& image, int size, int pad, bool pseudo) {
    CHECK_INPUT(image);

    torch::Tensor result = get_conv_empty(image, size, pad);
    median_filter_op(result, image, size, pad, pseudo);

    return result;
}
torch::Tensor bilateral_filter(const torch::Tensor& image, float std_k, float std_i, int size, int pad) {
    CHECK_INPUT(image);

    torch::Tensor result = get_conv_empty(image, size, pad);
    bilateral_filter_op(result, image, std_k, std_i, size, pad);

    return result;
}
torch::Tensor custom_conv(const torch::Tensor& image, const torch::Tensor& kernel, int pad) {
    CHECK_INPUT(image);
    CHECK_INPUT(kernel);

    int size = kernel.size(0) / 2;
    torch::Tensor result = get_conv_empty(image, size, pad);
    custom_conv_op(result, image, kernel.to(torch::kFloat), pad);

    return result;
}
torch::Tensor conditional_match(const torch::Tensor& image, const torch::Tensor& patterns) {
    CHECK_INPUT(image);
    CHECK_INPUT(patterns);

    torch::Tensor mark = torch::empty_like(image);
    conditional_match_op(mark, image, patterns);

    return mark;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uniform_conv", &uniform_conv, "Uniform Convolution");
    m.def("gaussian_conv", &gaussian_conv, "Gaussian Convolution");
    m.def("median_filter", &median_filter, "Median/Pseudo-Median Filter");
    m.def("bilateral_filter", &bilateral_filter, "Bilateral Filter");
    m.def("custom_conv", &custom_conv, "Convolution with Custom Kernel");
    m.def("conditional_match", &conditional_match, "Conditional Match the Pattern");
}