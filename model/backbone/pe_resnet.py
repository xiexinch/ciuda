import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models import BACKBONES, build_backbone
from mmseg.ops import resize
from mmseg.models.utils import PatchEmbed
from mmcv.cnn.utils.weight_init import trunc_normal_


@BACKBONES.register_module()
class PositionEncodingResNet(BaseModule):

    def __init__(self, img_size: tuple, patch_size: int, resnet_cfg: dict, init_cfg=None):
        super().__init__(init_cfg)
        self.img_size = img_size
        self.patch_size = patch_size

        h, w = self.img_size
        patch_h, patch_w = h // patch_size, w // patch_size

        self.pos_embed = nn.Parameter(torch.zeros(torch.Size((1, 3, patch_h, patch_w))), requires_grad=True)
        self.backbone = build_backbone(resnet_cfg)

        trunc_normal_(self.pos_embed)

    def forward(self, inputs):

        pos_emb = resize(self.pos_embed, size=inputs.shape[2:], mode='bicubic', align_corners=False)
        x = inputs + pos_emb
        return self.backbone(x)
