import os.path as osp

import torch
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel

from mmgen.models.builder import MODELS, build_module
from mmgen.models.misc import tensor2img
from mmgen.models.common import GANImageBuffer, set_requires_grad
from mmgen.models import BaseGAN

from mmseg.models import build_segmentor, build_loss
from mmseg.ops import resize


@MODELS.register_module()
class SegGAN(BaseGAN):
    """CycleGAN model for unpaired image-to-image translation.

    Ref:
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        cycle_loss (dict): Config for the cycle-consistency loss.
        id_loss (dict): Config for the identity loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generator
            update.
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
            `direction`: image-to-image translation direction (the model
            training direction): a2b | b2a.
            `buffer_size`: GAN image buffer size.
        test_cfg (dict): Config for testing. Default: None.
            You may change the testing of gan by setting:
            `direction`: image-to-image translation direction (the model
            training direction): a2b | b2a.
            `show_input`: whether to show input real images.
            `test_direction`: direction in the test mode (the model testing
            direction). CycleGAN has two generators. It decides whether
            to perform forward or backward translation with respect to
            `direction` during testing: a2b | b2a.
        pretrained (str): Path for pretrained model. Default: None.
    """
    def __init__(self,
                 segmentor,
                 discriminator,
                 gan_loss,
                 ce_loss,
                 static_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generators
        self.segmentors = nn.ModuleDict()
        self.segmentors['a'] = build_segmentor(segmentor)

        # discriminators
        self.discriminators = nn.ModuleDict()
        self.discriminators['a'] = build_module(discriminator)

        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        self.image_buffers['source'] = GANImageBuffer(self.buffer_size)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_module(gan_loss)
        self.ce_loss = build_loss(ce_loss)
        self.static_loss = build_loss(static_loss)

        # others
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.step_counter = 0  # counting training steps

        self.init_weights(pretrained)

        self.use_ema = False

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        self.segmentors['a'].init_weights()
        self.discriminators['a'].init_weights(pretrained)

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module

        return module

    def forward_train(self, img, label, is_source):
        """Forward function for training.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            dict: Dict of forward results for training.
        """

        segmentor = self.get_module(self.segmentors)['a']

        feat = segmentor.extract_feat(img)
        pred = segmentor._decode_head_forward_test(feat, dict())
        pred = resize(input=pred,
                      size=img.shape[2:],
                      mode='bilinear',
                      align_corners=False)

        results = dict(img=img,
                       seg_logits=pred,
                       feat=feat[-1],
                       label=label,
                       is_source=is_source)
        return results

    def forward_test(self, **kwargs):
        """Forward function for testing.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Dict of forward and evaluation results for testing.
        """

        segmentors = self.get_module(self.segmentors)

        return

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.

        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        segmentors = self.get_module(self.segmentors)
        return segmentors['a'].encode_decode(img, img_metas=dict())

    def forward(self,
                img,
                label=None,
                is_source=None,
                test_mode=False,
                **kwargs):
        """Forward function.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if not test_mode:
            return self.forward_train(
                img,
                label,
                is_source,
            )
        segmentor = self.get_module(self.segmentors)['a']
        return segmentor(return_loss=test_mode,
                         img=img,
                         img_metas=dict(),
                         **kwargs)

    def backward_discriminators(self, outputs):
        """Backward function for the discriminators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        discriminators = self.get_module(self.discriminators)

        log_vars_d = dict()

        losses = dict()
        # GAN loss for discriminators['a']
        pred = discriminators['a'](outputs['feat'].detach().contiguous())
        is_source = outputs['is_source'].data
        # losses['loss_gan_d'] = 0
        # for i, p in enumerate(pred):
        #     p = p.unsqueeze(0)
        #     losses['loss_gan_d'] += self.gan_loss(p,
        #                                           target_is_real=bool(
        #                                               is_source[i]),
        #                                           is_disc=True)

        losses['loss_gan_d'] = self.gan_loss(
            pred,
            target_is_real=outputs['is_source'].float().contiguous(),
            is_disc=True)

        loss_d_a, log_vars_d_a = self._parse_losses(losses)
        # loss_d_a *= 0.5
        loss_d_a.backward()
        log_vars_d['loss_gan_d'] = log_vars_d_a['loss']

        return log_vars_d

    def backward_generators(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        discriminator = self.get_module(self.discriminators)['a']

        losses = dict()

        # GAN loss for segmentors['a']
        # with torch.no_grad():
        #     pred = discriminator(outputs['feat'])
        pred = discriminator(outputs['feat'])
        is_source = outputs['is_source'].data

        losses['loss_gan_g'] = self.gan_loss(
            pred,
            target_is_real=outputs['is_source'].float().contiguous(),
            is_disc=False) * 0.001
        # Forward ce loss
        losses['loss_seg'] = 0
        count = 0
        for i, f in enumerate(is_source):
            pred = outputs['seg_logits'][i].unsqueeze(0)
            label = outputs['label'][i]
            if bool(f.numel()):
                losses['loss_seg'] += self.ce_loss(pred, label)
            else:
                losses['loss_seg'] += self.static_loss(pred, label)
            count += 1

        losses['loss_seg'] /= count

        # pred = outputs['seg_logits']
        # label = outputs['label'].squeeze(1)
        # losses['loss_static'] = self.static_loss(pred, label)
        # losses['loss_seg'] = self.ce_loss(pred, label)

        loss_g, log_vars_g = self._parse_losses(losses)
        loss_g.backward()

        return log_vars_g

    def train_step(self, data_batch, optimizer):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminators.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # data
        img = data_batch['img']
        label = data_batch['gt_semantic_seg']
        is_source = data_batch['is_source']

        # forward generators
        outputs = self.forward(img, label, is_source, test_mode=False)

        log_vars = dict()

        # discriminators
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()
        log_vars.update(self.backward_discriminators(outputs=outputs))
        optimizer['discriminators'].step()

        # generators, no updates to discriminator parameters.
        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['segmentors'].zero_grad()
            log_vars.update(self.backward_generators(outputs=outputs))
            optimizer['segmentors'].step()

        self.step_counter += 1

        segmentor = self.get_module(self.segmentors)['a']
        segmentor.PALETTE = PALETTE
        segmentor.CLASSES = CLASSES
        seg_source = segmentor.show_result(
            img.permute(0, 2, 3, 1).cpu().numpy()[0],
            outputs['seg_logits'].detach().cpu().numpy()[0], opacity=0.1)

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(log_vars=log_vars,
                       num_samples=len(outputs['img']),
                       results=dict(seg_pred=torch.from_numpy(
                           seg_source).permute(2, 0, 1).unsqueeze(0)))

        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.

        Returns:
            dict: Dict of evaluation results for validation.
        """
        # forward generator
        results = self.get_module(self.segmentors)['a'].val_step(data_batch)
        return results


CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
           [0, 0, 230], [119, 11, 32]]
