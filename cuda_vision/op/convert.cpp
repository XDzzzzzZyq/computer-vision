#include <torch/extension.h>

void to_grayscale_op(
    torch::Tensor& gray,
    const torch::Tensor& image
);
void invert_op(
    torch::Tensor& result,
    const torch::Tensor& image
);
void hash_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    int seed
);
void threshold_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    float threshold
);
void random_threshold_op(
    torch::Tensor& result,
    const torch::Tensor& image,
    const torch::Tensor& noise
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor to_grayscale(const torch::Tensor& image) {
    CHECK_INPUT(image);

    int b = image.size(0);
    int h = image.size(2);
    int w = image.size(3);
    torch::Tensor gray = torch::empty({b, 1, h, w}).to(image.device());
    to_grayscale_op(gray, image);

    return gray;
}

torch::Tensor invert(const torch::Tensor& image) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    invert_op(result, image);

    return result;
}

torch::Tensor hash(const torch::Tensor& image, int seed) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    hash_op(result, image, seed);

    return result;
}

torch::Tensor threshold(const torch::Tensor& image, float threshold) {
    CHECK_INPUT(image);

    torch::Tensor result = torch::empty_like(image);
    threshold_op(result, image, threshold);

    return result;
}

torch::Tensor random_threshold(const torch::Tensor& image, const torch::Tensor& noise) {
    CHECK_INPUT(image);
    CHECK_INPUT(noise);

    torch::Tensor result = torch::empty_like(image);
    random_threshold_op(result, image, noise);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_grayscale", &to_grayscale, "Convert RGB images to grayscales");
    m.def("invert", &invert, "Invert RGB color/grayscale of images");
    m.def("hash", &hash, "Return uniform distributed white noise");
    m.def("threshold", &threshold, "Binarize the given images with fixed Threshold");
    m.def("random_threshold", &random_threshold, "Random Binarize the given images by grayscale");
}