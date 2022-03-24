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
class SegGAN2(BaseGAN):
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

    def forward_train(self, img, label, img_day, img_night):

        segmentor = self.get_module(self.segmentors)['a']

        # with torch.no_grad():
        #     feat = segmentor.extract_feat(img)
        # pred = segmentor._decode_head_forward_test(feat, dict())

        feat = segmentor.extract_feat(img)
        pred = segmentor._decode_head_forward_test(feat, dict())
        pred = resize(input=pred,
                      size=img.shape[2:],
                      mode='bilinear',
                      align_corners=False)

        city_feat = segmentor.extract_feat(img)
        city_pred = segmentor._decode_head_forward_test(city_feat, dict())
        city_pred = resize(input=city_pred,
                           size=img.shape[2:],
                           mode='bilinear',
                           align_corners=False)

        results = dict(img=img, seg_logits=pred, feat=feat[-1], label=label)
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
                label,
                img_day,
                img_night,
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
            return self.forward_train(img, label, img_day, img_night)
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
        discriminators = self.get_module(self.discriminators)['a']

        log_vars_d = dict()

        losses = dict()

        source_pred = discriminators(
            outputs['source_seg_logits'].detach().contiguous())
        target_pred = discriminators(
            outputs['target_seg_logits'].detach().contiguous())

        losses['loss_gan_d_s'] = self.gan_loss(source_pred,
                                               target_is_real=True,
                                               is_disc=True)

        losses['loss_gan_d_t'] = self.gan_loss(target_pred,
                                               target_is_real=False,
                                               is_disc=True)

        loss_d_a, log_vars_d_a = self._parse_losses(losses)
        loss_d_a *= 0.5
        loss_d_a.backward()
        log_vars_d['loss_gan_d'] = log_vars_d_a['loss'] * 0.5

        return log_vars_d

    def forward_source(self, img, label):
        segmentor = self.get_module(self.segmentors)['a']

        feat = segmentor.extract_feat(img)
        pred = segmentor._decode_head_forward_test(feat, dict())
        pred = resize(input=pred,
                      size=img.shape[2:],
                      mode='bilinear',
                      align_corners=False)

        results = dict(seg_logits=pred, label=label)
        return results

    def forward_target(self, img_day, img_night):
        segmentor = self.get_module(self.segmentors)['a']

        day_pred = segmentor.encode_decode(img_day, dict())
        night_pred = segmentor.encode_decode(img_night, dict())

        psudo_prob = torch.zeros_like(day_pred)
        threshold = torch.ones_like(day_pred[:, :11, :, :]) * 0.2
        threshold[day_pred[:, :11, :, :] > 0.4] = 0.8
        psudo_prob[:, :11, :, :] = threshold * day_pred[:, :11, :, :].detach(
        ) + (1 - threshold) * night_pred[:, :11, :, :].detach()
        psudo_prob[:, 11:, :, :] = night_pred[:, 11:, :, :].detach()
        weights = torch.log(
            torch.FloatTensor([
                0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272,
                0.01227341, 0.00207795, 0.0055127, 0.15928651, 0.01157818,
                0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456,
                0.00235192, 0.00232904, 0.00098658, 0.00413907
            ])).cuda()
        weights = (torch.mean(weights) -
                   weights) / torch.std(weights) * 0.05 + 1.0

        weights_prob = weights.expand(psudo_prob.size()[0],
                                      psudo_prob.size()[3],
                                      psudo_prob.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        psudo_prob = psudo_prob * weights_prob
        pseudo_gt = torch.argmax(psudo_prob.detach(), dim=1)
        pseudo_gt[pseudo_gt >= 11] = 255

        results = dict(seg_logits=night_pred, label=pseudo_gt)
        return results

    def backward_segmentor(self, source_outputs, target_outputs):
        discriminators = self.get_module(self.discriminators)['a']

        losses = dict()
        pred = discriminators(
            target_outputs['seg_logits'].detach().contiguous())
        losses['loss_seg'] = self.ce_loss(source_outputs['seg_logits'],
                                          source_outputs['label'])
        losses['loss_gan_ss'] = self.gan_loss(pred,
                                              target_is_real=True,
                                              is_disc=False)
        pred = discriminators(
            target_outputs['seg_logits'].detach().contiguous())

        losses['loss_gan_st'] = self.gan_loss(pred,
                                              target_is_real=False,
                                              is_disc=False)

        losses['loss_static'] = self.static_loss(target_outputs['seg_logits'],
                                                 target_outputs['label'])
        loss, log_vars = self._parse_losses(losses)
        loss.backward()
        return log_vars

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
        img_day = data_batch['img_day']
        img_night = data_batch['img_night']

        log_vars = dict()

        source_outputs = self.forward_source(img, label)
        target_outputs = self.forward_target(img_day, img_night)

        # discriminators
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()

        log_vars.update(
            self.backward_discriminators(
                outputs=dict(source_seg_logits=source_outputs['seg_logits'],
                             target_seg_logits=target_outputs['seg_logits'])))
        optimizer['discriminators'].step()

        # generators, no updates to discriminator parameters.
        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['segmentors'].zero_grad()
            log_vars.update(
                self.backward_segmentor(source_outputs, target_outputs))
            optimizer['segmentors'].step()

        self.step_counter += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(log_vars=log_vars, results=dict())

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
