import filters
import combine
import torch


def get_gradient_kernel(type='pixel_difference'):
    if type == 'pixel_diff':
        row = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        col = row.transpose(1, 0).contiguous()
    elif type == 'sep_pixel_diff':
        row = torch.tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        col = row.transpose(1, 0).contiguous()
    elif type == 'roberts':
        row = torch.tensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
        col = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif type == 'prewitt':
        row = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
        col = row.transpose(1, 0).contiguous()
    elif type == 'sobel':
        row = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        col = row.transpose(1, 0).contiguous()
    else:
        row = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        col = row.clone()

    return row.float().cuda(), col.float().cuda()


def gradient_conv(img: torch.Tensor, type):
    row_kernel, col_kernel = get_gradient_kernel(type)
    grad_x = filters.custom_conv(img, row_kernel, 1)
    grad_y = filters.custom_conv(img, col_kernel, 1)
    return grad_x, grad_y
