"""
Emgraph loss functions package
"""
from .absolute_margin import AbsoluteMarginLoss
from .bce import BCELoss
from .nll import NLLLoss
from .nll_multiclass import NLLMulticlass
from .pairwise import PairwiseLoss
from .self_adversarial import SelfAdversarialLoss
from ._loss_constants import LOSS_REGISTRY

__all__ = ['AbsoluteMarginLoss', 'BCELoss', 'NLLLoss', 'NLLMulticlass', 'PairwiseLoss', 'SelfAdversarialLoss',
           'LOSS_REGISTRY']