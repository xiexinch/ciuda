import torch


class PolyLrUpdater(object):
    def __init__(self, base_lr, max_iters, power=1., min_lr=0.):

        super(PolyLrUpdater, self).__init__()
        self.base_lr = base_lr
        self.power = power
        self.min_lr = min_lr
        self.max_iters = max_iters

    def set_lr(self, optimizer: torch.optim.Optimizer, current_iter: int):

        coeff = (1 - current_iter / self.max_iters)**self.power
        lr = (self.base_lr - self.min_lr) * coeff + self.min_lr
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10
