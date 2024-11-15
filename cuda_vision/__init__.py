if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.convert import matrix_dither

    img = load_raw("../imgs/barbara.raw", 256, 256, 1)
    dit1 = matrix_dither(img, 2)
    dit2 = matrix_dither(img, 4)

    compare_imgs([img, dit1, dit2])
    plt.show()
