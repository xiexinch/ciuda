from typing import Iterable, List
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import NECKS

@NECKS.register_module()
class PositionEncodeNeck(nn.Module):

    def __init__(self, in_channels: Iterable[int]) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
