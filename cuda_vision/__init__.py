if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.__kernels import get_dither_matrix

    index1 = get_dither_matrix(2)
    index2 = get_dither_matrix(4)
    print(index1)
    print(index2)
