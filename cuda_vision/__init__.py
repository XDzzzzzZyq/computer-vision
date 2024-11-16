import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.filters import morphology
    from cuda_vision.convert import invert

    img = load_raw("../imgs/pcb.raw", 256, 256, 1)
    img = invert(img)
    o = morphology(img, 'o')
    c = morphology(img, 'c')
    compare_imgs([img, o, c])
    plt.show()
