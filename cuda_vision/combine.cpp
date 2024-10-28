#include <torch/extension.h>
#include <vector>

void watermark_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& mark,
    int offset_x, int offset_y
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor watermark(const torch::Tensor& image, const torch::Tensor& mark, int offset_x, int offset_y) {
    CHECK_INPUT(image);
    CHECK_INPUT(mark);

    torch::Tensor result = image.clone();
    watermark_op(result, image, mark, offset_x, offset_y);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("watermark", &watermark, "Overlay the Watermark to Target Image");
}