import numpy as np
import torch
import matplotlib.pyplot as plt


def load_raw(path: str, h, w, c, dtype=np.uint8, device='cuda') -> torch.Tensor:
    """
    :param path:   Path of .raw
    :param w:      Specify the width
    :param h:      Specify the height
    :param c:      Specify the channel
    :param dtype:  Specify the pixle type
    :param device: Specify the device
    :return:
        Tensor with shape (B, C, W, H)
    """

    image = np.fromfile(path, dtype=dtype)
    image = torch.from_numpy(image)
    image = image.reshape(h, w, c)
    image = image.transpose(0, 2).unsqueeze(0).contiguous()
    return image.to(device).float()


def load_png(path: str, device='cuda'):
    import imageio.v2 as imageio
    image = imageio.imread(path)
    image = torch.from_numpy(image)
    image = image.transpose(0, 2).unsqueeze(0).contiguous()
    if image.shape[1] == 4:
        image = image[:, 0:3].contiguous()
    return image.to(device).float()


def show_img(image, range=(0.0, 255.0), interpolation='nearest', ax=plt):
    if image.ndim == 4:
        image = image[0]
    range = range if range is not None else (None, None)
    ax.imshow(image.transpose(0, 2).detach().cpu().int(), cmap='gray', vmin=range[0], vmax=range[1],
              interpolation=interpolation)


def compare_imgs(images: list, range=(0.0, 255.0), interpolation='nearest'):
    n = len(images)
    fig, axe = plt.subplots(nrows=1, ncols=n, figsize=(10 * n, 10))
    for i, ax in enumerate(axe):
        show_img(images[i], range=range, interpolation=interpolation, ax=ax)


def compare_imgs_grid(images: list, shape, range=(0.0, 255.0), interpolation='nearest', transpose=False):
    assert shape[0]*shape[1] == len(images)
    fig, axe = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(10 * shape[1], 10 * shape[0]))
    for i, axr in enumerate(axe):
        for j, ax in enumerate(axr):
            idx = j*shape[0]+i if transpose else i*shape[1]+j
            show_img(images[idx], range=range, interpolation=interpolation, ax=ax)


def show_hist(image, bins=256, range=(0, 255), ax=plt):
    if image.ndim == 4:
        image = image[0]
    ax.hist(image.flatten().detach().cpu().int(), bins=bins, range=range, color='black', alpha=0.7)


def show_accumulative_hist(image, bins=256, range=(0, 255), ax=plt):
    if image.ndim == 4:
        image = image[0]
    hist_counts, bin_edges = torch.histogram(image.flatten().detach().cpu(), bins=bins, range=range)
    cumulative_counts = torch.cumsum(hist_counts, dim=0)
    width = (bin_edges[1] - bin_edges[0]).item()
    ax.bar(bin_edges[1:], cumulative_counts, width=width, align='edge', color='black', alpha=0.7)


def compare_hist(images: list, bins=256, range=(0, 255), accu=False):
    n = len(images)
    hist = show_accumulative_hist if accu else show_hist
    fig, axe = plt.subplots(nrows=1, ncols=n, figsize=(12 * n, 10))
    for i, ax in enumerate(axe):
        hist(images[i], bins=bins, range=range, ax=ax)


if __name__ == '__main__':
    texs = [load_raw(f'../imgs/sample{i}.raw', 64, 64, 1) for i in range(1, 16)]
    compare_imgs_grid(texs, shape=(3, 5), transpose=True)
    plt.show()
