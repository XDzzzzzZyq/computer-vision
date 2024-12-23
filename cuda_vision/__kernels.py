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
    elif type == 'gaussian3':
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 4
    elif type == 'gaussian5':
        lap = torch.tensor([[0, 0, 1, 0, 0],
                            [0, 1, 2, 1, 0],
                            [1, 2,-16,2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 0, 1, 0, 0]]) / 16
    else:
        lap = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    return lap.float().cuda()


S1 = torch.Tensor([
    [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
S2 = torch.Tensor([
    [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]]])
S3 = torch.Tensor([
    [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
    [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 1], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 1], [0, 1, 0]]])
TK4 = torch.Tensor([
    [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 1, 1], [0, 1, 0]]])
STK4 = torch.Tensor([
    [[0, 0, 1], [0, 1, 1], [0, 0, 1]],
    [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 1]]])
ST5 = torch.Tensor([
    [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
    [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
    [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
    [[0, 0, 0], [0, 1, 1], [0, 1, 1]]])
ST6 = torch.Tensor([
    [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    [[0, 1, 1], [1, 1, 0], [1, 0, 0]]])
STK6 = torch.Tensor([
    [[1, 1, 1], [0, 1, 1], [0, 0, 0]],
    [[1, 0, 0], [1, 1, 0], [1, 1, 0]],
    [[0, 0, 0], [0, 1, 1], [1, 1, 1]],
    [[0, 0, 1], [0, 1, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
    [[1, 1, 0], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [1, 1, 1]],
    [[0, 1, 1], [0, 1, 1], [0, 0, 1]]])
STK7 = torch.Tensor([
    [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
    [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 0], [1, 1, 1]]])
STK8 = torch.Tensor([
    [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
    [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
    [[0, 0, 0], [1, 1, 1], [1, 1, 1]]])
STK9 = torch.Tensor([
    [[1, 1, 1], [0, 1, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [1, 1, 0], [1, 1, 0]],
    [[1, 0, 0], [1, 1, 1], [1, 1, 1]],
    [[0, 1, 1], [0, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [1, 1, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1], [1, 1, 1]]])
STK10 = torch.Tensor([
    [[1, 1, 1], [0, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 0, 1]],
    [[1, 1, 1], [1, 1, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 1, 1], [1, 1, 1]]])
K11 = torch.Tensor([
    [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[1, 1, 0], [1, 1, 1], [1, 1, 1]],
    [[0, 1, 1], [1, 1, 1], [1, 1, 1]]])


def get_conditional_patterns(type):
    if type in ['S', 's', 'shrink']:
        p = torch.cat([S1, S2, S3, STK4, ST5, ST6, STK6, STK7, STK8, STK9, STK10])
    elif type in ['T', 't', 'thin']:
        p = torch.cat([TK4, STK4, ST5, ST6, STK6, STK7, STK8, STK9, STK10])
    elif type in ['K', 'k', 'skeletonize']:
        p = torch.cat([TK4, STK4, STK6, STK7, STK8, STK9, STK10, K11])
    else:
        p = torch.ones(1, 3, 3)

    return p

# Spur
Sp1 = torch.Tensor([
    [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 0]]])
Sp2 = torch.Tensor([
    [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
# Single 4-connection
S4c1 = torch.Tensor([
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 1], [0, 0, 0]]])
S4c2 = torch.Tensor([
    [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [0, 0, 0]]])
# L Corner
LC1 = torch.Tensor([
    [[0, 0, 1], [0, 1, 1], [0, 0, 0]], 
    [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
    [[1, 1, 0], [0, 1, 0], [0, 0, 0]], 
    [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [1, 0, 0]], 
    [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]], 
    [[0, 0, 0], [0, 1, 1], [0, 0, 1]]])
LC2 = torch.Tensor([
    [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 1, 0]]])
# 4-connected Offset
c4O = torch.Tensor([
    [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
    [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 1], [0, 1, 0]]])
# Spur corner Cluster
ScC = torch.Tensor([
    [[0, -2, 1], [0, 1, -2], [1, 0, 0]],
    [[1, -2, 0], [-2, 1, 0], [0, 0, 1]],
    [[0, 0, 1], [-2, 1, 0], [1, -2, 0]],
    [[1, 0, 0], [0, 1, -2], [0, -2, 1]]])
# Corner Cluster
CC1 = torch.Tensor([
    [[1, 1, -1], [1, 1, -1], [-1, -1, -1]]])
CC2 = torch.Tensor([
    [[-1, -1, -1], [-1, 1, 1], [-1, 1, 1]]])
# Tee Branch
TB1 = torch.Tensor([
    [[-1, 1, 0], [1, 1, 1], [-1, 0, 0]], 
    [[0, 1, -1], [1, 1, 1], [0, 0, -1]],
    [[0, 0, -1], [1, 1, 1], [0, 1, -1]], 
    [[-1, 0, 0], [1, 1, 1], [-1, 1, 0]],
    [[-1, 1, -1], [1, 1, 0], [0, 1, 0]], 
    [[0, 1, 0], [1, 1, 0], [-1, 1, -1]],
    [[0, 1, 0], [0, 1, 1], [-1, 1, -1]], 
    [[-1, 1, -1], [0, 1, 1], [0, 1, 0]]])
TB2 = torch.Tensor([
    [[-1, 1, -1], [1, 1, 1], [-1, -1, -1]],
    [[-1, -1, -1], [1, 1, 1], [-1, 1, -1]],
    [[-1, 1, -1], [1, 1, -1], [-1, 1, -1]],
    [[-1, 1, -1], [-1, 1, 1], [-1, 1, -1]]])
# Vee Branch
VB = torch.Tensor([
    [[1, -1,  1], [-1, 1, -1], [-2,-2, -2]],
    [[1, -1, -2], [-1, 1, -2], [1, -1, -2]],
    [[-2, -2,-2], [-1, 1, -1], [1, -1,  1]],
    [[-2, -1, 1], [-2, 1, -1], [-2, -1, 1]]])
DB = torch.Tensor([
    [[-1, 1, 0], [0, 1, 1], [1, 0, -1]], 
    [[0, 1, -1], [1, 1, 0], [-1, 0, 1]],
    [[-1, 0, 1], [1, 1, 0], [0, 1, -1]], 
    [[1, 0, -1], [0, 1, 1], [-1, 1, 0]]])


def get_unconditional_patterns(type):
    if type in ['S', 's', 'shrink']:
        p = torch.cat([Sp1, S4c1, c4O, ScC, CC1, TB1, VB, DB])
    elif type in ['T', 't', 'thin']:
        p = torch.cat([Sp1, S4c1, LC1, c4O, ScC, CC1, TB1, VB, DB])
    elif type in ['K', 'k', 'skeletonize']:
        p = torch.cat([Sp1, Sp2, S4c1, S4c2, LC2, ScC, CC1, CC2, TB2, VB, DB])
    else:
        p = torch.ones(1, 3, 3)

    return p


def get_dither_matrix(n):
    if n == 2:
        index = torch.Tensor([[1, 2], [3, 0]])
    else:
        index1 = get_dither_matrix(n // 2) * 4 + 1
        index2 = get_dither_matrix(n // 2) * 4 + 2
        index3 = get_dither_matrix(n // 2) * 4 + 3
        index4 = get_dither_matrix(n // 2) * 4 + 0
        col1, col2 = torch.cat([index1, index3]), torch.cat([index2, index4])
        index = torch.cat([col1, col2], dim=1)

    return index


def get_diffuse_matrix(type):
    if type == 'floyd-steinberg':
        m = torch.Tensor([[0, 0, 0], [0, 0, 7], [3, 5, 1]])/16
    elif type == 'jarvis-judice-ninke':
        m = torch.Tensor([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 7, 5],
                          [3, 5, 7, 5, 3],
                          [1, 3, 5, 3, 1]]) / 48
    else:
        m = torch.ones(3, 3)
    return m


L3laws = torch.tensor([1,  2, 1])/6
E3laws = torch.tensor([-1, 0, 1])/2
S3laws = torch.tensor([1, -2, 1])/2


def get_laws_kernel():
    v = torch.stack([L3laws, E3laws, S3laws], dim=0)
    outer = torch.einsum('ik,jl->ijkl', v, v)
    return outer.reshape(9, 3, 3)


def get_metric_patterns():
    Q = torch.tensor(
        [[[0, 0], [0, 0]],
         [[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]],
         [[1, 1], [0, 0]], [[0, 1], [0, 1]], [[0, 0], [1, 1]], [[1, 0], [1, 0]],
         [[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]], [[1, 1], [1, 0]],
         [[1, 1], [1, 1]],
         [[1, 0], [0, 1]], [[0, 1], [1, 0]]]).float()
    return Q
