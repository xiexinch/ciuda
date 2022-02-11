from mmcv.runner import BaseModule
from mmseg.models import BACKBONES, build_backbone

from ..core import CIConv2d


@BACKBONES.register_module()
class CIResNet(BaseModule):

    def __init__(self, invariant, resnet_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.invariant = invariant
        self.ciconv = CIConv2d(self.invariant, kernel_size=3, scale=2.0)
        self.backbone = build_backbone(resnet_cfg)

    def forward(self, inputs):
        x = self.ciconv(inputs)
        return self.backbone(x)
