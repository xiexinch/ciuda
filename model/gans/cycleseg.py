import os.path as osp

import mmcv
import numpy as np
import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint

from mmgen.models.builder import MODELS, build_module
from mmgen.models.misc import tensor2img
from mmgen.models.common import GANImageBuffer, set_requires_grad
from mmgen.models import BaseGAN

from mmseg.models import build_segmentor, build_loss as build_seg_loss


@MODELS.register_module()
class CycleSeg(BaseGAN):
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
                 generator,
                 discriminator,
                 segmentor_d,
                 segmentor_n,
                 gan_loss,
                 cycle_loss,
                 ce_loss,
                 id_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pretrained_seg_d=None,
                 pretrained_seg_n=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # identity loss only works when input and output images have the same
        # number of channels
        if id_loss is not None and id_loss.get('loss_weight') > 0.0:
            assert generator.get('in_channels') == generator.get(
                'out_channels')

        # generators
        self.generators = nn.ModuleDict()
        self.generators['a'] = build_module(generator)
        self.generators['b'] = build_module(generator)

        # discriminators
        self.discriminators = nn.ModuleDict()
        self.discriminators['a'] = build_module(discriminator)
        self.discriminators['b'] = build_module(discriminator)

        # segmentors
        self.segmentor_d = build_segmentor(segmentor_d)
        self.segmentor_n = build_segmentor(segmentor_n)

        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        self.image_buffers['a'] = GANImageBuffer(self.buffer_size)
        self.image_buffers['b'] = GANImageBuffer(self.buffer_size)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_module(gan_loss)
        assert cycle_loss is not None  # cycle loss cannot be None
        self.cycle_loss = build_module(cycle_loss)
        self.id_loss = build_module(id_loss) if id_loss else None
        self.ce_loss = build_seg_loss(ce_loss)

        # others
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        if self.train_cfg is None:
            self.direction = ('a2b' if self.test_cfg is None else
                              self.test_cfg.get('direction', 'a2b'))
        else:
            self.direction = self.train_cfg.get('direction', 'a2b')
        self.step_counter = 0  # counting training steps
        self.show_input = (False if self.test_cfg is None else
                           self.test_cfg.get('show_input', False))
        # In CycleGAN, if not showing input, we can decide the translation
        # direction in the test mode, i.e., whether to output fake_night or fake_day
        if not self.show_input:
            self.test_direction = ('a2b' if self.test_cfg is None else
                                   self.test_cfg.get('test_direction', 'a2b'))
            if self.direction == 'b2a':
                self.test_direction = ('b2a' if self.test_direction == 'a2b'
                                       else 'a2b')

        self.pretrained_seg_d = pretrained_seg_d
        self.pretrained_seg_n = pretrained_seg_n
        self.init_weights(pretrained)

        self.use_ema = False

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        self.generators['a'].init_weights(pretrained=pretrained)
        self.generators['b'].init_weights(pretrained=pretrained)
        self.discriminators['a'].init_weights(pretrained=pretrained)
        self.discriminators['b'].init_weights(pretrained=pretrained)

        load_checkpoint(self.segmentor_d, self.pretrained_seg_d)
        load_checkpoint(self.segmentor_n, self.pretrained_seg_n)

        # Fix teacher net
        set_requires_grad(self.segmentor_d, False)

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



    def forward_train(self, img_day, img_night, meta):
        """Forward function for training.

        Args:
            img_day (Tensor): Input image from domain A.
            img_night (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            dict: Dict of forward results for training.
        """


        generators = self.get_module(self.generators)

        fake_night = generators['a'](img_day)
        rec_day = generators['b'](fake_night)
        fake_day = generators['b'](img_night)
        rec_night = generators['a'](fake_day)

        seg_pred_day = self.segmentor_d.encode_decode(img_day)
        seg_pred_day_f = self.segmentor_d.encode_decode(fake_day)
        seg_pred_night = self.segmentor_n.encode_decode(img_night)
        seg_pred_night_f = self.segmentor_n.encode_decode(fake_night)

        results = dict(real_day=img_day,
                       fake_night=fake_night,
                       rec_day=rec_day,
                       real_night=img_night,
                       fake_day=fake_day,
                       rec_night=rec_night,
                       seg_pred_day=seg_pred_day,
                       seg_pred_day_f=seg_pred_day_f,
                       seg_pred_night=seg_pred_night,
                       seg_pred_night_f=seg_pred_night_f
                       )
        return results

    def forward_test(self,
                     img_day,
                     img_night,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Forward function for testing.

        Args:
            img_day (Tensor): Input image from domain A.
            img_night (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Dict of forward and evaluation results for testing.
        """
        # No need for metrics during training for CycleGAN. And
        # this is a special trick in CycleGAN original paper & implementation,
        # collecting the statistics of the test batch at test time.
        # In fact, no effects: IN + no dropout for CycleGAN.
        self.train()

        # necessary setup
        real_day, real_night, image_path = self.setup(img_day, img_night, meta)

        generators = self.get_module(self.generators)

        fake_night = generators['a'](img_day)
        fake_day = generators['b'](img_night)
        results = dict(real_day=real_day.cpu(),
                       fake_night=fake_night.cpu(),
                       real_night=real_night.cpu(),
                       fake_day=fake_day.cpu())

        # save image
        if save_image:
            assert save_path is not None
            folder_name = osp.splitext(osp.basename(image_path[0]))[0]
            if self.show_input:
                if iteration:
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{folder_name}-{iteration + 1:06d}-ra-fb-rb-fa.png')
                else:
                    save_path = osp.join(save_path,
                                         f'{folder_name}-ra-fb-rb-fa.png')
                output = np.concatenate([
                    tensor2img(results['real_day'], min_max=(-1, 1)),
                    tensor2img(results['fake_night'], min_max=(-1, 1)),
                    tensor2img(results['real_night'], min_max=(-1, 1)),
                    tensor2img(results['fake_day'], min_max=(-1, 1))
                ],
                                        axis=1)
            else:
                if self.test_direction == 'a2b':
                    if iteration:
                        save_path = osp.join(
                            save_path, folder_name,
                            f'{folder_name}-{iteration + 1:06d}-fb.png')
                    else:
                        save_path = osp.join(save_path,
                                             f'{folder_name}-fb.png')
                    output = tensor2img(results['fake_night'], min_max=(-1, 1))
                else:
                    if iteration:
                        save_path = osp.join(
                            save_path, folder_name,
                            f'{folder_name}-{iteration + 1:06d}-fa.png')
                    else:
                        save_path = osp.join(save_path,
                                             f'{folder_name}-fa.png')
                    output = tensor2img(results['fake_day'], min_max=(-1, 1))
            flag = mmcv.imwrite(output, save_path)
            results['saved_flag'] = flag

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.

        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        generators = self.get_module(self.generators)
        tmp = generators['a'](img)
        out = generators['b'](tmp)
        return out

    def forward(self, img_day, img_night, meta, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_day (Tensor): Input image from domain A.
            img_night (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if not test_mode:
            return self.forward_train(img_day, img_night, meta)

        return self.forward_test(img_day, img_night, meta, **kwargs)

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
        fake_night = self.image_buffers['b'].query(outputs['fake_night'])
        fake_pred = discriminators['a'](fake_night.detach())
        losses['loss_gan_d_a_fake'] = self.gan_loss(fake_pred,
                                                    target_is_real=False,
                                                    is_disc=True)
        real_pred = discriminators['a'](outputs['real_night'])
        losses['loss_gan_d_a_real'] = self.gan_loss(real_pred,
                                                    target_is_real=True,
                                                    is_disc=True)

        loss_d_a, log_vars_d_a = self._parse_losses(losses)
        loss_d_a *= 0.5
        loss_d_a.backward()
        log_vars_d['loss_gan_d_a'] = log_vars_d_a['loss'] * 0.5

        losses = dict()
        # GAN loss for discriminators['b']
        fake_day = self.image_buffers['a'].query(outputs['fake_day'])
        fake_pred = discriminators['b'](fake_day.detach())
        losses['loss_gan_d_b_fake'] = self.gan_loss(fake_pred,
                                                    target_is_real=False,
                                                    is_disc=True)
        real_pred = discriminators['b'](outputs['real_day'])
        losses['loss_gan_d_b_real'] = self.gan_loss(real_pred,
                                                    target_is_real=True,
                                                    is_disc=True)

        loss_d_b, log_vars_d_b = self._parse_losses(losses)
        loss_d_b *= 0.5
        loss_d_b.backward()
        log_vars_d['loss_gan_d_b'] = log_vars_d_b['loss'] * 0.5

        return log_vars_d

    def backward_generators(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        generators = self.get_module(self.generators)
        discriminators = self.get_module(self.discriminators)

        losses = dict()
        # Identity losses for generators
        if self.id_loss is not None and self.id_loss.loss_weight > 0:
            id_a = generators['a'](outputs['real_night'])
            losses['loss_id_a'] = self.id_loss(
                id_a, outputs['real_night']) * self.cycle_loss.loss_weight
            id_b = generators['b'](outputs['real_day'])
            losses['loss_id_b'] = self.id_loss(
                id_b, outputs['real_day']) * self.cycle_loss.loss_weight

        # GAN loss for generators['a']
        fake_pred = discriminators['a'](outputs['fake_night'])
        losses['loss_gan_g_a'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        # GAN loss for generators['b']
        fake_pred = discriminators['b'](outputs['fake_day'])
        losses['loss_gan_g_b'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        # Backward cycle loss
        losses['loss_cycle_a'] = self.cycle_loss(outputs['rec_a'],
                                                 outputs['real_day'])
        # Backward cycle loss
        losses['loss_cycle_b'] = self.cycle_loss(outputs['rec_b'],
                                                 outputs['real_night'])

        # # CE loss for segmentor_n
        # losses['loss_seg_d'] = self.ce_loss(outputs['seg_pred_night_f'], outputs['seg_pred_day'])
        # losses['loss_seg_n'] = self.ce_loss(outputs['seg_pred_night'], outputs['seg_pred_day_f'])

        loss_g, log_vars_g = self._parse_losses(losses)
        loss_g.backward()

        return log_vars_g
    
    def backward_segmentor(self, outputs):
        
        losses = dict()
        losses['loss_seg_d'] = self.ce_loss(outputs['seg_pred_night_f'], outputs['seg_pred_day'])
        losses['loss_seg_n'] = self.ce_loss(outputs['seg_pred_night'], outputs['seg_pred_day_f'])
        loss_seg, log_vars_seg = self._parse_losses(losses)
        loss_seg.backward()

        return log_vars_seg


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
        img_day = data_batch['img_day']
        img_night = data_batch['img_night']
        meta = data_batch['meta']

        # forward generators
        outputs = self.forward(img_day, img_night, meta, test_mode=False)

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
            set_requires_grad(self.segmentor_n, False)
            set_requires_grad(self.generators, True)
            # optimize generators
            optimizer['generators'].zero_grad()
            log_vars.update(self.backward_generators(outputs=outputs))
            optimizer['generators'].step()

            # optimize segmentor
            set_requires_grad(self.generators, False)
            set_requires_grad(self.segmentor_n, True)
            optimizer['segmentor_n'].zero_grad()
            log_vars.update(self.backward_segmentor(outputs=outputs))
            optimizer['segmentor_n'].step()


        self.step_counter += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(log_vars=log_vars,
                       num_samples=len(outputs['real_day']),
                       results=dict(real_day=outputs['real_day'].cpu(),
                                    fake_night=outputs['fake_night'].cpu(),
                                    real_night=outputs['real_night'].cpu(),
                                    fake_day=outputs['fake_day'].cpu()))

        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.

        Returns:
            dict: Dict of evaluation results for validation.
        """
        # data
        img_day = data_batch['img_day']
        img_night = data_batch['img_night']
        meta = data_batch['meta']

        # forward generator
        results = self.forward(img_day, img_night, meta, test_mode=True, **kwargs)
        return results
