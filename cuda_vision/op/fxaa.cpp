#include <torch/extension.h>
#include <iostream>

void mark_edge_op(
    torch::Tensor& result,
    const torch::Tensor& gray,
    const torch::Tensor& grad_x,
    const torch::Tensor& grad_y,
    float thres_min, float thres_max
);
void resample_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& edge,
    float r
);
void smooth_offset_op(
    torch::Tensor& result,
    const torch::Tensor& gray,
    const torch::Tensor& edge,
    int max_iter
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mark_edge(
    const torch::Tensor& gray,
    const torch::Tensor& grad_x,
    const torch::Tensor& grad_y,
    float thres_min, float thres_max
) {
    CHECK_INPUT(gray);
    CHECK_INPUT(grad_x);
    CHECK_INPUT(grad_y);

    int B = gray.size(0);
    int W = gray.size(2);
    int H = gray.size(3);

    torch::Tensor result = torch::empty({B, 2, W, H}).to(gray.device());
    mark_edge_op(result, gray, grad_x, grad_y, thres_min, thres_max);

    return result;
}

torch::Tensor resample(
    const torch::Tensor& image,
    const torch::Tensor& edge,
    float r
) {
    CHECK_INPUT(image);
    CHECK_INPUT(edge);

    torch::Tensor result = torch::empty_like(image);
    resample_op(result, image, edge, r);

    return result;
}

torch::Tensor smooth_offset(
    const torch::Tensor& gray,
    const torch::Tensor& edge,
    int max_iter
) {
    CHECK_INPUT(gray);
    CHECK_INPUT(edge);

    torch::Tensor result = torch::empty_like(edge);
    smooth_offset_op(result, gray, edge, max_iter);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mark_edge", &mark_edge, "Mark the edge to filter");
    m.def("resample", &resample, "Resample the original to finish");
    m.def("smooth_offset", &smooth_offset, "Smooth uv offset");
}