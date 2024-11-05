if __name__ == "__main__":
    from utils.imageIO import *
    from filters import uniform_conv, custom_conv

    ori = load_raw('../imgs/rose_color.raw', 256, 256, 3)
    noi = load_raw('../imgs/rose_color_noise.raw', 256, 256, 3)
    noi_r, noi_g, noi_b = noi[:, 0:1], noi[:, 1:2], noi[:, 2:3]
    noi1 = uniform_conv(noi_r, 2, 2)

    kernel1 = torch.ones(5).to(noi_r.device) / 5.0
    kernel2 = torch.ones(5, 5).to(noi_r.device) / 25.0

    noi2 = custom_conv(noi_r, kernel1, 2)
    noi3 = custom_conv(noi_r, kernel2, 2)

    compare_imgs([noi1, noi2, noi3, noi_r])
    #compare_imgs([filtered0, filtered4])
    plt.show()
