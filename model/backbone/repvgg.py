import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmseg.models import BACKBONES

from ..core import SEBlock, CIConv2d


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_ll = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, ratio=16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_ll,
                                   groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight /
              ((self.rbr_dense.bn.running_var +
                self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var +
                                         self.rbr_1x1.bn.eps).sqrt())).reshape(
                                             -1, 1, 1, 1).detach()

        # The L2 loss of the "circle" of weights in 3x3 kernel.
        # Use regular L2 on them.
        l2_loss_circle = (K3**2).sum() - (K3[:, :, 1:2, 1:2]**2).sum()
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel**2 / (t3**2 + t1**2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    #  This func derives the equivalent kernel and bias in a DIFFERENTIABLE
    #  way. You can get the equivalent kernel and bias at any time and do
    #  whatever you want, for example, apply some penalties or constraints
    #  during training, just like you do to the other models.
    #  May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


@BACKBONES.register_module()
class RepVGG(nn.Module):
    def __init__(self,
                 num_blocks,
                 width_multiplier=None,
                 override_groups_map=None,
                 deploy=False,
                 use_se=False,
                 invariant='E'):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        self.invariant = invariant

        if self.invariant:
            self.ciconv = CIConv2d(self.invariant, kernel_size=3, scale=2.0)

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3 if not self.invariant else 1,
                                  out_channels=self.in_planes,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  deploy=self.deploy,
                                  use_se=self.use_se)

        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]),
                                       num_blocks[0],
                                       stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]),
                                       num_blocks[1],
                                       stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]),
                                       num_blocks[2],
                                       stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]),
                                       num_blocks[3],
                                       stride=2)
        self.init_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=cur_groups,
                            deploy=self.deploy,
                            use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        if hasattr(self, 'ciconv'):
            x = self.ciconv(inputs)
            # from PIL import Image
            # img = x.clone()[0]
            # blue = np.array([67, 95, 233])
            # red = np.array([233, 30, 30])

            # img = img.detach().squeeze(0).cpu().numpy()
            # print(img.shape)
            # img_ = np.zeros(img.shape)
            # print(img_.shape)
            # img_ = img_[:, :, np.newaxis]
            # img_ = np.tile(img_, 3)
            # print(img_.shape)
            # img_[img > 0] = red
            # img_[img <= 0] = blue
            # img = Image.fromarray(img_.astype(np.uint8))

            # img.show()
            # raise '123'
            x = self.stage0(x)
        else:
            x = self.stage0(inputs)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return tuple([x1, x2, x3, x4])
