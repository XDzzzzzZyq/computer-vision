if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision import filters, convert

    pat = load_raw("../imgs/patterns.raw", 256, 256, 1)
    pat = convert.invert(pat)
    a = filters.conditional_match(pat, 's')
    compare_imgs([pat, a])
    plt.show()

