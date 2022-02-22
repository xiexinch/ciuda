from .adv_loss import least_square_loss
from .static_loss import StaticLoss
from .cx_loss import ContexturalLoss, ContextualBilateralLoss

__all__ = ['least_square_loss', 'StaticLoss', 'ContexturalLoss', 'ContextualBilateralLoss']
