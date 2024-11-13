if __name__ == "__main__":
    from utils.imageIO import *
    from detect import gradient_conv, gradient_edge_detect, laplacian_edge_detect
    from convert import binarize

    ori = load_raw('../imgs/building.raw', 256, 256, 1)
    orn = load_raw('../imgs/building_noise.raw', 256, 256, 1)

    edge0 = laplacian_edge_detect(ori, type='prewitt', threshold=10, mode=0)
    edge1 = laplacian_edge_detect(ori, type='prewitt', threshold=10, mode=1)
    edge2 = laplacian_edge_detect(ori, type='prewitt', threshold=15, mode=1)
    edge3 = laplacian_edge_detect(ori, type='gaussian3', threshold=10, mode=0)
    edge4 = laplacian_edge_detect(ori, type='gaussian5', threshold=10, mode=0)
    compare_imgs([ori, edge0, edge1, edge2, edge3, edge4])

    plt.show()
