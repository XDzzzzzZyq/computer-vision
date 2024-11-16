import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.filters import morphology
    from cuda_vision.convert import invert

    img = load_raw("../imgs/pcb.raw", 256, 256, 1)
    img = invert(img)
    d = morphology(img, 'd')
    e = morphology(img, 'e')
    compare_imgs([img, d, e])
    plt.show()
