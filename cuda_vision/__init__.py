if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision import filters, convert

    img = load_raw("../imgs/barbara.raw", 256, 256, 1)
    h1 = convert.hash(img, 1)
    h2 = convert.hash(img, 2)
    h3 = convert.hash(img, 3)
    compare_imgs([img, h1, h3])
    compare_hist([h1, h2, h3])
    plt.show()



