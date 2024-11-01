#include <torch/extension.h>

void overlay_op(
    torch::Tensor& result,
    const torch::Tensor& a,
    const torch::Tensor& b,
    int code
);
void watermark_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& mark,
    int offset_x, int offset_y
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor overlay(const torch::Tensor& a, const torch::Tensor& b, int code) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    torch::Tensor result = a.clone();
    overlay_op(result, a, b, code);

    return result;
}

torch::Tensor watermark(const torch::Tensor& image, const torch::Tensor& mark, int offset_x, int offset_y) {
    CHECK_INPUT(image);
    CHECK_INPUT(mark);

    torch::Tensor result = image.clone();
    watermark_op(result, image, mark, offset_x, offset_y);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("overlay", &overlay, "Overlay the Images to Target Images by the Given Operation");
    m.def("watermark", &watermark, "Overlay the Watermark to Target Images");
}