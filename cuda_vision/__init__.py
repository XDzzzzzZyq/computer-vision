import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.convert import error_diffusion

    img = load_raw("../imgs/barbara.raw", 256, 256, 1)
    dit1 = error_diffusion(img, type='floyd-steinberg', threshold=80.)
    dit2 = error_diffusion(img, type='floyd-steinberg', threshold=127.5)
    dit3 = error_diffusion(img, type='floyd-steinberg', threshold=170.)
    compare_imgs([img, dit1, dit2, dit3])
    plt.show()
