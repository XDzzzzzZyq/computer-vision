if __name__ == "__main__":
    from utils.imageIO import *
    from cuda_vision import filters, convert

    if True:
        pat = load_raw("../imgs/patterns.raw", 256, 256, 1)
        pat = convert.invert(pat)

        l = [pat]
        m = [pat]
        c = [pat]

        for i in range(10):
            post, mark, m_cp = filters.morphology(l[i], 's')
            l.append(post)
            m.append(mark)
            c.append(m_cp)
        compare_imgs([i[:, :, 83:90, 183:190] for i in l[3:5]])
        compare_imgs([i[:, :, 83:90, 183:190] for i in m[3:5]])
        compare_imgs([i[:, :, 83:90, 183:190] for i in c[3:5]])
        plt.show()
    else:
        pat = torch.tensor([[0, 255, 0, 0, 0], [0, 255, 0, 0, 0], [255, 255, 255, 255, 0], [255, 255, 255, 255, 255],
                            [0, 255, 0, 0, 0], [0, 255, 0, 0, 0]]).reshape(1, 1, 6, 5).cuda().float()
        pat = pat.transpose(3, 2).contiguous()

        l = [pat]

        for i in range(2):
            g, mark, m_cp = filters.morphology(l[i], 's')
            l.append(post)
        compare_imgs(l)
        plt.show()


