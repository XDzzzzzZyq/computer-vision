if __name__ == "__main__":
    from utils.imageIO import *
    from detect import gradient_conv
    from convert import binarize

    ori = load_raw('../imgs/building.raw', 256, 256, 1)
    orn = load_raw('../imgs/building_noise.raw', 256, 256, 1)

    gx, gy = gradient_conv(ori, type='prewitt')
    gx_b = binarize((gx+gy).abs(), 20.0)

    compare_imgs([ori, gx, gy, gx_b])
    plt.show()
