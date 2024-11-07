if __name__ == "__main__":
    from utils.imageIO import *
    from detect import gradient_conv, gradient_edge_detect
    from convert import binarize

    ori = load_raw('../imgs/building.raw', 256, 256, 1)
    orn = load_raw('../imgs/building_noise.raw', 256, 256, 1)

    gx, gy = gradient_conv(ori, type='prewitt')
    edge = gradient_edge_detect(ori, 30, type='prewitt')

    compare_imgs([ori, gx, gy, edge])
    plt.show()
