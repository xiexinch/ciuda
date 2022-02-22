import os.path as osp

import mmcv
import numpy as np
import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel

from mmgen.models.builder import MODELS, build_module
from mmgen.models.misc import tensor2img
from mmgen.models.common import GANImageBuffer, set_requires_grad
from mmgen.models import BaseGAN


@MODELS.register_module()
class RoundGANV2(BaseGAN):
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
                 gan_loss,
                 cycle_loss,
                 cx_loss=None,
                 id_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
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
        self.generators['c'] = build_module(generator)
        self.generators['d'] = build_module(generator)

        # discriminators
        self.discriminators = nn.ModuleDict()
        self.discriminators['a'] = build_module(discriminator)
        self.discriminators['b'] = build_module(discriminator)
        self.discriminators['c'] = build_module(discriminator)

        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        self.image_buffers['a'] = GANImageBuffer(self.buffer_size)
        self.image_buffers['b'] = GANImageBuffer(self.buffer_size)
        self.image_buffers['c'] = GANImageBuffer(self.buffer_size)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_module(gan_loss)
        assert cycle_loss is not None  # cycle loss cannot be None
        self.cycle_loss = build_module(cycle_loss)
        self.cx_loss = build_module(cx_loss) if cx_loss else None
        self.id_loss = build_module(id_loss) if id_loss else None
        

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
        # direction in the test mode, i.e., whether to output fake_b or fake_a
        if not self.show_input:
            self.test_direction = ('a2b' if self.test_cfg is None else
                                   self.test_cfg.get('test_direction', 'a2b'))
            if self.direction == 'b2a':
                self.test_direction = ('b2a' if self.test_direction == 'a2b'
                                       else 'a2b')

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
        self.generators['c'].init_weights(pretrained=pretrained)
        self.generators['d'].init_weights(pretrained=pretrained)
        self.discriminators['a'].init_weights(pretrained=pretrained)
        self.discriminators['b'].init_weights(pretrained=pretrained)
        self.discriminators['c'].init_weights(pretrained=pretrained)

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

    def setup(self, img_a, img_b, img_c, meta):
        """Perform necessary pre-processing steps.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            img_c (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            Tensor, Tensor, list[str]: The real images from domain A/B, and \
                the image path as the metadata.
        """
        a2b = self.direction == 'a2b'
        real_a = img_a if a2b else img_b
        real_b = img_b if a2b else img_a
        real_c = img_c
        image_path = [v['img_a_path' if a2b else 'img_b_path'] for v in meta]

        return real_a, real_b, real_c, image_path

    def forward_train(self, img_a, img_b, img_c, meta):
        """Forward function for training.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            dict: Dict of forward results for training.
        """
        # necessary setup
        real_a, real_b, real_c, _ = self.setup(img_a, img_b, img_c, meta)

        generators = self.get_module(self.generators)

        fake_b = generators['a'](real_a)
        rec_a = generators['b'](fake_b)
        fake_a = generators['b'](real_b)
        rec_b = generators['a'](fake_a)

        fake_c = generators['c'](real_b)
        rec_b_ = generators['d'](fake_c)
        fake_b_ = generators['d'](real_c)
        rec_c = generators['c'](fake_b_)

        fake_c_ = generators['c'](fake_b)
        rec_a_ = generators['b'](generators['d'](fake_c_))
        fake_a_ = generators['b'](fake_b_)
        rec_c_ = generators['c'](generators['a'](fake_a_))

        results = dict(real_a=real_a,
                       fake_b=fake_b,
                       rec_a=rec_a,
                       real_b=real_b,
                       fake_a=fake_a,
                       rec_b=rec_b,
                       real_c=real_c,
                       fake_b_=fake_b_,
                       rec_c=rec_c,
                       rec_b_=rec_b_,
                       fake_c=fake_c,
                       fake_c_=fake_c_,
                       rec_a_=rec_a_,
                       fake_a_=fake_a_,
                       rec_c_=rec_c_)
        return results

    def forward_test(self,
                     img_a,
                     img_b,
                     img_c,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
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
        # No need for metrics during training for CycleGAN. And
        # this is a special trick in CycleGAN original paper & implementation,
        # collecting the statistics of the test batch at test time.
        # In fact, no effects: IN + no dropout for CycleGAN.
        self.train()

        # necessary setup
        real_a, _, _, image_path = self.setup(img_a, img_b, img_c, meta)

        generators = self.get_module(self.generators)
        fake_b = generators['a'](real_a)
        fake_c = generators['c'](fake_b)


        results = dict(real_a=real_a.cpu(),
                       fake_b=fake_b.cpu(),
                       fake_c=fake_c.cpu())

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
                    tensor2img(results['real_a'], min_max=(-1, 1)),
                    tensor2img(results['fake_b'], min_max=(-1, 1)),
                    tensor2img(results['fake_c'], min_max=(-1, 1)),
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
                    output = tensor2img(results['fake_c'], min_max=(-1, 1))
                else:
                    pass
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
        tmp = generators['b'](tmp)
        out = generators['c'](tmp)
        return out

    def forward(self, img_a, img_b, img_c, meta, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if not test_mode:
            return self.forward_train(img_a, img_b, img_c, meta)

        return self.forward_test(img_a, img_b, img_c, meta, **kwargs)

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
        fake_a = self.image_buffers['a'].query(outputs['fake_a'])
        fake_pred = discriminators['a'](fake_a.detach())
        losses['loss_gan_d_a_fake'] = self.gan_loss(fake_pred,
                                                    target_is_real=False,
                                                    is_disc=True)
        real_pred = discriminators['a'](outputs['real_a'])
        losses['loss_gan_d_a_real'] = self.gan_loss(real_pred,
                                                    target_is_real=True,
                                                    is_disc=True)
        fake_a_ = self.image_buffers['a'].query(outputs['fake_a_'])
        fake_pred_ = discriminators['a'](fake_a_.detach())
        losses['loss_gan_d_a1_fake'] = self.gan_loss(fake_pred_,
                                                     target_is_real=False,
                                                     is_disc=True)

        loss_d_a, log_vars_d_a = self._parse_losses(losses)
        loss_d_a *= 0.5
        loss_d_a.backward()
        log_vars_d['loss_gan_d_a'] = log_vars_d_a['loss'] * 0.5

        losses = dict()
        # GAN loss for discriminators['b']
        fake_b = self.image_buffers['b'].query(outputs['fake_b'])
        fake_pred = discriminators['b'](fake_b.detach())
        losses['loss_gan_d_b_fake'] = self.gan_loss(fake_pred,
                                                    target_is_real=False,
                                                    is_disc=True)
        real_pred = discriminators['b'](outputs['real_b'])
        losses['loss_gan_d_b_real'] = self.gan_loss(real_pred,
                                                    target_is_real=True,
                                                    is_disc=True)
        fake_b_ = self.image_buffers['b'].query(outputs['fake_b_'])
        fake_pred_ = discriminators['b'](fake_b_.detach())
        losses['loss_gan_d_b1_fake'] = self.gan_loss(fake_pred_,
                                                     target_is_real=False,
                                                     is_disc=True)

        loss_d_b, log_vars_d_b = self._parse_losses(losses)
        loss_d_b *= 0.5
        loss_d_b.backward()
        log_vars_d['loss_gan_d_b'] = log_vars_d_b['loss'] * 0.5

        losses = dict()
        # GAN loss for discriminators['c']
        fake_c = self.image_buffers['c'].query(outputs['fake_c'])
        fake_pred = discriminators['c'](fake_c.detach())
        losses['loss_gan_d_c_fake'] = self.gan_loss(fake_pred,
                                                    target_is_real=False,
                                                    is_disc=True)
        real_pred = discriminators['c'](outputs['real_c'])
        losses['loss_gan_d_c_real'] = self.gan_loss(real_pred,
                                                    target_is_real=True,
                                                    is_disc=True)
        fake_c_ = self.image_buffers['c'].query(outputs['fake_c_'])
        fake_pred_ = discriminators['c'](fake_c_.detach())
        losses['loss_gan_d_c1_fake'] = self.gan_loss(fake_pred_,
                                                     target_is_real=False,
                                                     is_disc=True)

        loss_d_c, log_vars_d_c = self._parse_losses(losses)
        loss_d_c *= 0.5
        loss_d_c.backward()
        log_vars_d['loss_gan_d_c'] = log_vars_d_c['loss'] * 0.5

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
            id_a = generators['a'](outputs['real_b'])
            losses['loss_id_a'] = self.id_loss(
                id_a, outputs['real_b']) * self.cycle_loss.loss_weight
            id_b = generators['b'](outputs['real_a'])
            losses['loss_id_b'] = self.id_loss(
                id_b, outputs['real_a']) * self.cycle_loss.loss_weight
            id_c = generators['c'](outputs['real_c'])
            losses['loss_id_c'] = self.id_loss(
                id_c, outputs['real_c']) * self.cycle_loss.loss_weight
            id_d = generators['d'](outputs['real_b'])
            losses['loss_id_d'] = self.id_loss(
                id_d, outputs['real_b']) * self.cycle_loss.loss_weight

        # GAN loss for generators['a']
        fake_pred = discriminators['b'](outputs['fake_b'])
        losses['loss_gan_g_b'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        # GAN loss for generators['b']
        fake_pred = discriminators['a'](outputs['fake_a'])
        losses['loss_gan_g_a'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        fake_pred = discriminators['a'](outputs['fake_a_'])
        losses['loss_gan_g_a_'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        # GAN loss for generators['c']
        fake_pred = discriminators['c'](outputs['fake_c'])
        losses['loss_gan_g_c'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)
        fake_pred = discriminators['c'](outputs['fake_c_'])
        losses['loss_gan_g_c_'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)

        # GAN loss for generators['d']
        fake_pred = discriminators['b'](outputs['fake_b_'])
        losses['loss_gan_g_d_'] = self.gan_loss(fake_pred,
                                               target_is_real=True,
                                               is_disc=False)

        # Forward cycle loss
        losses['loss_cycle_a'] = self.cycle_loss(outputs['rec_a'],
                                                 outputs['real_a'])
        # Backward cycle loss
        losses['loss_cycle_b'] = self.cycle_loss(outputs['rec_b'],
                                                 outputs['real_b'])
        # Backward cycle loss
        losses['loss_cycle_c'] = self.cycle_loss(outputs['rec_c'],
                                                 outputs['real_c'])            
        # Forward cycle loss
        losses['loss_cycle_a_'] = self.cycle_loss(outputs['rec_a_'],
                                                 outputs['real_a'])
        # Backward cycle loss
        losses['loss_cycle_b_'] = self.cycle_loss(outputs['rec_b_'],
                                                 outputs['real_b'])
        # Backward cycle loss
        losses['loss_cycle_c_'] = self.cycle_loss(outputs['rec_c_'],
                                                 outputs['real_c'])
                                            
        if self.cx_loss is not None:
            losses['cx_loss_a'] = self.cx_loss(outputs['real_a'], outputs['fake_a'])
            losses['cx_loss_b'] = self.cx_loss(outputs['real_b'], outputs['fake_b'])
            losses['cx_loss_c'] = self.cx_loss(outputs['real_c'], outputs['fake_c'])
            losses['cx_loss_a_'] = self.cx_loss(outputs['real_a_'], outputs['fake_a_'])
            losses['cx_loss_b_'] = self.cx_loss(outputs['real_b_'], outputs['fake_b_'])
            losses['cx_loss_c_'] = self.cx_loss(outputs['real_c_'], outputs['fake_c_'])

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
        img_a = data_batch['img_a']
        img_b = data_batch['img_b']
        img_c = data_batch['img_c']
        meta = data_batch['meta']

        # forward generators
        outputs = self.forward(img_a, img_b, img_c, meta, test_mode=False)

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
            optimizer['generators'].zero_grad()
            log_vars.update(self.backward_generators(outputs=outputs))
            optimizer['generators'].step()

        self.step_counter += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(log_vars=log_vars,
                       num_samples=len(outputs['real_a']),
                       results=dict(real_a=outputs['real_a'].cpu(),
                                    fake_b=outputs['fake_b'].cpu(),
                                    real_b=outputs['real_b'].cpu(),
                                    fake_a=outputs['fake_a'].cpu(),
                                    real_c=outputs['real_c'].cpu(),
                                    fake_c=outputs['fake_c'].cpu(),
                                    fake_c_=outputs['fake_c_'].cpu(),
                                    fake_b_=outputs['fake_b_'].cpu(),
                                    fake_a_=outputs['fake_a_'].cpu()))

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
        img_a = data_batch['img_a']
        img_b = data_batch['img_b']
        img_c = data_batch['img_c']
        meta = data_batch['meta']

        # forward generator
        results = self.forward(img_a, img_b, img_c, meta, test_mode=True, **kwargs)
        return results
