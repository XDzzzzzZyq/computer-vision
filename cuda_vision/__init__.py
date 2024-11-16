import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision.convert import threshold, error_diffusion, error_diffusion_fast

    img = load_raw("../imgs/barbara.raw", 256, 256, 1)
    dit0 = threshold(img, threshold=127.5)
    dit1 = error_diffusion(img, type='floyd-steinberg', thres=127.5)
    dit3 = error_diffusion_fast(img, type='floyd-steinberg', thres=127.5, step=3, relaxation=0.2)
    dit4 = error_diffusion_fast(img, type='floyd-steinberg', thres=127.5, step=5, relaxation=0.1)
    dit5 = error_diffusion_fast(img, type='floyd-steinberg', thres=127.5, step=10, relaxation=0.1)
    compare_imgs([dit0, dit1, dit3, dit4, dit5])
    plt.show()
