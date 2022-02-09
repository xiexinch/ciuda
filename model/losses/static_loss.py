import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticLoss(nn.Module):
    def __init__(self,
                 weights: torch.Tensor,
                 num_classes=19,
                 gamma=1.0,
                 eps=1e-7,
                 size_average=True,
                 one_hot=True,
                 ignore_index=255):
        super(StaticLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore_index = ignore_index
        self.weights = weights
        self.raw = False
        if (num_classes < 19):
            self.raw = True

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, eps=1e-5):
        B, C, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)

        if self.raw:
            target_left = target
            target_right = target
            target_up = target
            target_down = target

            target_left[:, :-1, :] = target[:, 1:, :]
            target_right[:, 1:, :] = target[:, :-1, :]
            target_up[:, :, 1:] = target[:, :, :-1]
            target_down[:, :, :-1] = target[:, :, 1:]

            target_left = target_left.view(-1)
            target_right = target_right.view(-1)
            target_up = target_up.view(-1)
            target_down = target_down.view(-1)

        assert target.shape == torch.Size((B, H, W))
        target = target.view(-1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            x = x[valid]
            target = target[valid]
            if self.raw:
                target_left = target_left[valid]
                target_right = target_right[valid]
                target_up = target_up[valid]
                target_down = target_down[valid]

        if self.one_hot:
            target_onehot = F.one_hot(target, x.size(1))
            if self.raw:
                left_right = F.one_hot(target_left, C) + F.one_hot(
                    target_right, C)
                up_down = F.one_hot(target_up, C) + F.one_hot(target_down, C)
                left_right2 = left_right.clone()
                up_down2 = up_down.clone()
                target_onehot2 = left_right + up_down + left_right2 + up_down2

                target_onehot = target_onehot + target_onehot2
                target_onehot[target_onehot > 1] = 1

        probs = F.softmax(x, dim=1)
        probs = (self.weights * probs * target_onehot).max(1)[0]
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow(1 - probs, self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
