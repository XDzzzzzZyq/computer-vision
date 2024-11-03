import torch


def search_param1d(target, func, p_range, steps, criterion=torch.nn.MSELoss()):
    error = []
    p_space = torch.linspace(*p_range, steps=steps)
    for p in p_space:
        pred = func(p)
        e = criterion(target, pred)
        error.append(e.cpu().item())
    error = torch.tensor(error)
    return p_space, error


def search_param2d(target, func, p_range, q_range, steps, criterion=torch.nn.MSELoss()):
    error = []
    p_space = torch.linspace(*p_range, steps=steps)
    q_space = torch.linspace(*q_range, steps=steps)
    for p in p_space:
        e_p = []
        for q in q_space:
            pred = func(p, q)
            e = criterion(target, pred)
            e_p.append(e.cpu().item())
        error.append(e_p)
    error = torch.tensor(error)
    return p_space, q_space, error


def optimal1d(p_space, error):
    return p_space[torch.argmin(error)]


def optimal2d(p_space, q_space, error):
    idx = torch.argmin(error)
    idx = torch.unravel_index(idx, error.shape)
    return p_space[idx[0]], q_space[idx[1]]


if __name__ == '__main__':
    def f2(x, y):
        return x**2 + y**2

    def f1(x):
        return f2(x, 0)

    p_space, error = search_param1d(torch.tensor([0]), f1, (-1,1), 11)
    opt = optimal1d(p_space, error)
    print(opt)

    p_space, q_space, error = search_param2d(torch.tensor([0]), f2, (-1, 1), (-1, 1), 11)
    opt = optimal2d(p_space, q_space, error)
    print(opt)