import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models import BACKBONES, build_backbone
from mmseg.ops import resize

def pos_size(img_size):
    h, w = img_size
    return tuple((1, 3, h, w))

@BACKBONES.register_module()
class PositionEncodingResNet(BaseModule):

    def __init__(self, img_size: tuple, resnet_cfg: dict, init_cfg=None):
        super().__init__(init_cfg)
        self.img_size = img_size
        self.pos_embed = nn.Parameter(torch.zeros(torch.Size(pos_size(self.img_size))))
        self.backbone = build_backbone(resnet_cfg)

    def forward(self, inputs):

        if inputs.shape[2:] != self.pos_embed.shape[2:]:
            pos_emb = resize(self.pos_embed, size=inputs.shape[2:], mode='bicubic', align_corners=False)
        else:
            pos_emb = self.pos_embed
        x = inputs + pos_emb
        return self.backbone(x)
