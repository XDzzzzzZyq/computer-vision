import numpy as np
import torch
import matplotlib.pyplot as plt


def load_raw(path: str, w, h, c, dtype=np.uint8, device='cuda') -> torch.Tensor:
    """
    :param path:   Path of .raw
    :param w:      Specify the width
    :param h:      Specify the height
    :param c:      Specify the channel
    :param dtype:  Specify the pixle type
    :param device: Specify the device
    :return:
        Tensor with shape (B, C, H, W)
    """

    image = np.fromfile(path, dtype=dtype)
    image = torch.from_numpy(image)
    image = image.reshape(w, h, c)
    image = image.transpose(0, 2).unsqueeze(0).contiguous()
    return image.to(device).float()


def show_img(image, interpolation='nearest', ax=plt):
    if image.ndim == 3:
        ax.imshow(image.transpose(0, 2).detach().cpu().int(), interpolation=interpolation)
    else:
        ax.imshow(image[0].transpose(0, 2).detach().cpu().int(), cmap='gray', interpolation=interpolation)


def compare_imgs(images:list, interpolation='nearest'):
    n = len(images)
    fig, axe = plt.subplots(nrows=1, ncols=n, figsize=(25 * n, 25))
    for i, ax in enumerate(axe):
        show_img(images[i], interpolation, ax)


if __name__ == '__main__':
    image = load_raw('../imgs/dku_logo_color.raw', 128, 128, 3)
    show_img(image)
    plt.show()
