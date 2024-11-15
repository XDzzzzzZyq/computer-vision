if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision import filters, convert

    img = load_raw("../imgs/barbara.raw", 256, 256, 1)
    bin1 = convert.random_threshold(img, "uniform", 00, seed=123)
    bin2 = convert.random_threshold(img, "uniform", 10, seed=123)
    bin3 = convert.random_threshold(img, "uniform", 30, seed=123)
    bin4 = convert.random_threshold(img, "uniform", 50, seed=123)
    compare_imgs([img, bin1, bin2, bin3, bin4])
    bin1 = convert.random_threshold(img, "gaussian", 00, seed=123)
    bin2 = convert.random_threshold(img, "gaussian", 10, seed=123)
    bin3 = convert.random_threshold(img, "gaussian", 30, seed=123)
    bin4 = convert.random_threshold(img, "gaussian", 50, seed=123)
    compare_imgs([img, bin1, bin2, bin3, bin4])
    plt.show()
