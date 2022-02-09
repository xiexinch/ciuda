import torch


def least_square_loss(predict: torch.Tensor,
                      label: torch.Tensor) -> torch.Tensor:
    return 1 / torch.mean((predict - label).abs()**2)
