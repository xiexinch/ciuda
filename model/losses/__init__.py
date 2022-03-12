from .adv_loss import least_square_loss
from .static_loss import StaticLoss
from .cx_loss import ContexturalLoss, ContextualBilateralLoss
from .perceptual_loss import PerceptualLoss
from .batch_gan_loss import BatchGANLoss

__all__ = [
    'least_square_loss', 'StaticLoss', 'ContexturalLoss',
    'ContextualBilateralLoss', 'PerceptualLoss', 'BatchGANLoss'
]
