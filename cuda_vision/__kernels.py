import torch


def get_gradient_kernel(type='pixel_diff'):
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


def get_laplacian_kernel(type='pixel_diff'):
    if type == 'pixel_diff':
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4
    elif type == 'prewitt':
        lap = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
    else:
        lap = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    return lap.float().cuda()