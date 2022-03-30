from operator import mod
from mmcv.runner.base_module import BaseModule
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init, constant_init, normal_init, build_upsample_layer
from torch.nn.modules import padding

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


class ChannelAttention(BaseModule):

    def __init__(self, channels, reduction=16, init_cfg=None):
        super(ChannelAttention, self).__init__(init_cfg)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(), nn.Conv2d(channels // reduction,
                                 channels,
                                 1,
                                 bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max_pool = self.max_pool(x)
        x_avg_pool = self.avg_pool(x)
        x_max = self.se(x_max_pool)
        x_avg = self.se(x_avg_pool)
        out = self.sigmoid(x_max + x_avg)
        return out


class SpatialAttention(BaseModule):

    def __init__(self, kernel_size=7, init_cfg=None):
        super(SpatialAttention, self).__init__(init_cfg)
        self.conv = ConvModule(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               act_cfg=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_max, x_avg], dim=1)

        out = self.conv(x_cat)
        out = self.sigmoid(out)
        return out


class BilateralAttn(BaseModule):

    def __init__(self, channels, reduction, kernel_size, init_cfg=None):
        super(BilateralAttn, self).__init__(init_cfg)
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m.weight, mode='fan_out')
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)
            elif isinstance(m, nn.Linear):
                normal_init(m.weight, std=0.001)
                if m.bias is not None:
                    constant_init(m.bias, 0)

    def forward(self, x):
        residual = x.clone()
        print(self.ca(x).shape)
        print(self.sa(x).shape)
        from torchvision.utils import make_grid
        import cv2
        import numpy as np
        sa_attn = self.sa(x)
        sa_attn = make_grid(sa_attn[0], normalize=True)
        sa_img = sa_attn.permute(1, 2, 0).detach().cpu().numpy()
        sa_img = (sa_img * 255).astype(np.uint8)
        cv2.imwrite('work_dirs/sa_attn.png', sa_img)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x + residual


@HEADS.register_module()
class FCCMHead(BaseDecodeHead):

    def __init__(self,
                 in_channels=(32, 320),
                 channels=352,
                 norm_cfg=dict(type='BN'),
                 with_fuse_attn=False,
                 **kwargs):

        super(FCCMHead, self).__init__(in_channels, channels, **kwargs)

        # 滤波器
        self.up_conv = ConvModule(in_channels[-1],
                                  in_channels[-1],
                                  3,
                                  padding=1,
                                  norm_cfg=norm_cfg)

        # spatial attn
        self.spatial_attn = BilateralAttn(in_channels[0],
                                          reduction=16,
                                          kernel_size=7)

        # semantic attn
        self.semantic_attn = BilateralAttn(in_channels[-1],
                                           reduction=16,
                                           kernel_size=7)

        # fuse attn
        self.with_fuse_attn = with_fuse_attn
        if self.with_fuse_attn:
            self.fuse_attn = BilateralAttn(sum(in_channels),
                                           reduction=32,
                                           kernel_size=7)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        # 先模糊再上采样
        x_32 = self.up_conv(x_32)
        x_32 = resize(x_32,
                      scale_factor=4,
                      mode='bilinear',
                      align_corners=True)
        x_32 = self.semantic_attn(x_32)

        # 空间注意力
        x_8 = self.spatial_attn(x_8)

        x_cat = torch.cat([x_8, x_32], dim=1)
        if self.with_fuse_attn:
            x_cat = self.fuse_attn(x_cat)

        # A*context + B*spatial + bias
        out = self.conv_seg(x_cat)
        return out


@HEADS.register_module()
class FCCMHead_EXT(BaseDecodeHead):

    def __init__(self,
                 in_channels=(32, 320),
                 channels=352,
                 norm_cfg=dict(type='BN'),
                 with_fuse_attn=False,
                 **kwargs):

        super(FCCMHead_EXT, self).__init__(in_channels, channels, **kwargs)

        # 滤波器
        self.up_conv = ConvModule(in_channels[-1],
                                  in_channels[-1],
                                  3,
                                  padding=1,
                                  norm_cfg=norm_cfg)

        # spatial attn
        self.spatial_attn = BilateralAttn(in_channels[0],
                                          reduction=16,
                                          kernel_size=7)

        # semantic attn
        self.semantic_attn = BilateralAttn(in_channels[-1],
                                           reduction=16,
                                           kernel_size=7)

        # fuse attn
        self.with_fuse_attn = with_fuse_attn
        if self.with_fuse_attn:
            self.fuse_attn = ConvModule(in_channels=sum(in_channels),
                                        out_channels=channels,
                                        kernel_size=3,
                                        padding=1,
                                        norm_cfg=norm_cfg)

        # learnable upsampling
        self.upsample = build_upsample_layer(
            dict(type='carafe', channels=in_channels[-1], scale_factor=4))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        # 先模糊再上采样
        x_32 = self.up_conv(x_32)
        x_32 = self.upsample(x_32)
        x_32 = self.semantic_attn(x_32)

        # 空间注意力
        x_8 = self.spatial_attn(x_8)

        x_cat = torch.cat([x_8, x_32], dim=1)
        if self.with_fuse_attn:
            x_cat = self.fuse_attn(x_cat)

        # A*context + B*spatial + bias
        out = self.conv_seg(x_cat)
        return out


@HEADS.register_module()
class FFCCMHead_EXT(BaseDecodeHead):

    def __init__(self,
                 in_channels=(32, 320),
                 channels=352,
                 norm_cfg=dict(type='BN'),
                 with_fuse_attn=True,
                 **kwargs):

        super(FFCCMHead_EXT, self).__init__(in_channels, channels, **kwargs)

        # 滤波器
        self.up_conv = ConvModule(in_channels[-1],
                                  in_channels[-1],
                                  3,
                                  padding=1,
                                  norm_cfg=norm_cfg)

        # # spatial attn
        # self.spatial_attn = BilateralAttn(
        #     in_channels[0], reduction=16, kernel_size=7)

        # # semantic attn
        # self.semantic_attn = BilateralAttn(
        #     in_channels[-1], reduction=16, kernel_size=7)

        # fuse attn
        self.with_fuse_attn = with_fuse_attn
        if self.with_fuse_attn:
            self.fuse_attn = BilateralAttn(sum(in_channels),
                                           reduction=32,
                                           kernel_size=7)

        # learnable upsampling
        self.upsample = build_upsample_layer(
            dict(type='carafe', channels=in_channels[-1], scale_factor=4))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        # 先模糊再上采样
        x_32 = self.up_conv(x_32)
        x_32 = self.upsample(x_32)
        # x_32 = self.semantic_attn(x_32)

        # 空间注意力
        # x_8 = self.spatial_attn(x_8)

        x_cat = torch.cat([x_8, x_32], dim=1)
        if self.with_fuse_attn:
            x_cat = self.fuse_attn(x_cat)

        # A*context + B*spatial + bias
        out = self.conv_seg(x_cat)
        return out


@HEADS.register_module()
class BiFCCMHead_EXT(BaseDecodeHead):

    def __init__(self,
                 in_channels=(32, 320),
                 channels=352,
                 norm_cfg=dict(type='BN'),
                 with_fuse_attn=False,
                 **kwargs):

        super(BiFCCMHead_EXT, self).__init__(in_channels, channels, **kwargs)

        self.semantic_up = nn.Sequential(
            ConvModule(in_channels[-1],
                       in_channels[0],
                       3,
                       padding=1,
                       norm_cfg=norm_cfg),
            build_upsample_layer(
                dict(type='carafe', channels=in_channels[0], scale_factor=4)))

        self.context_down = nn.Sequential(
            ConvModule(in_channels[0],
                       in_channels[0],
                       3,
                       padding=1,
                       norm_cfg=norm_cfg),
            ConvModule(in_channels[0],
                       in_channels[-1],
                       3,
                       padding=1,
                       stride=2,
                       norm_cfg=norm_cfg),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        self.upsample = build_upsample_layer(
            dict(type='carafe', channels=in_channels[-1], scale_factor=4))
        self.activate = nn.Sigmoid()

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x_8, x_32 = x[0], x[1]

        x_32_up = self.semantic_up(x_32)
        x_8_down = self.context_down(x_8)

        x_32 = x_32 * self.activate(x_8_down)
        x_32 = self.upsample(x_32)

        x_8 = x_8 * self.activate(x_32_up)

        x_cat = torch.cat([x_8, x_32], dim=1)

        # A*context + B*spatial + bias
        out = self.conv_seg(x_cat)
        return out
