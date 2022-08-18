import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import build_segmentor, SEGMENTORS, BaseSegmentor

@SEGMENTORS.register_module()
class DistilledSegmentor(BaseSegmentor):
    
    def __init__(self,
                 segmentor_d,
                 segmentor_n,
                 pretrained_segmentor_d=None,
                 pretrained_segmentor_n=None,
                 init_cfg=None):
        super(DistilledSegmentor, self).__init__(init_cfg)
        self.segmentor_d = build_segmentor(segmentor_d)
        self.segmentor_n = build_segmentor(segmentor_n)
        self.pretrained_segmentor_d = pretrained_segmentor_d
        self.pretrained_segmentor_n = pretrained_segmentor_n

        self.init_weights()
    
    def init_weights(self):
        super().init_weights()
        if self.pretrained_segmentor_d is not None:
            load_checkpoint(self.segmentor_d, self.pretrained_segmentor_d)
        if self.pretrained_segmentor_n is not None:
            load_checkpoint(self.segmentor_n, self.pretrained_segmentor_n)
    
    def forward_train(self):
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.segmentor_n.forward_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        return self.segmentor_n.simple_test(img, img_meta, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        return self.segmentor_n.aug_test(imgs, img_metas, **kwargs)
    
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train()
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self.segmentor_n(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs
    
    def train_step(self, data_batch, optimizer, **kwargs):
        img_source, label_source, img_target = data_batch['img_source'], data_batch['label_source'], data_batch['img_target']
        losses = self.segmentor_d(img_source)