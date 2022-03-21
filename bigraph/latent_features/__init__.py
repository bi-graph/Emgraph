# todo: add more models from here https://github.com/Sujit-O/pykg2vec/tree/master/pykg2vec/models


from .models.EmbeddingModel import EmbeddingModel, MODEL_REGISTRY, set_entity_threshold, reset_entity_threshold
from .models.TransE import TransE
from .models.DistMult import DistMult
from .models.ComplEx import ComplEx
from .models.HolE import HolE
from .models.RandomBaseline import RandomBaseline
from .models.ConvKB import ConvKB
from .models.ConvE import ConvE

from .loss_functions import Loss, AbsoluteMarginLoss, SelfAdversarialLoss, NLLLoss, PairwiseLoss, \
    NLLMulticlass, BCELoss, LOSS_REGISTRY
from .regularizers import Regularizer, LPRegularizer, REGULARIZER_REGISTRY
from .optimizers import Optimizer, AdagradOptimizer, AdamOptimizer, MomentumOptimizer, SGDOptimizer, OPTIMIZER_REGISTRY
from .initializers import Initializer, RandomNormal, RandomUniform, Xavier, Constant, INITIALIZER_REGISTRY
from .misc import get_entity_triples
from ..utils import save_model, restore_model

__all__ = ['LOSS_REGISTRY', 'REGULARIZER_REGISTRY', 'MODEL_REGISTRY', 'OPTIMIZER_REGISTRY', 'INITIALIZER_REGISTRY',
           'set_entity_threshold', 'reset_entity_threshold',
           'EmbeddingModel', 'TransE', 'DistMult', 'ComplEx', 'HolE', 'ConvKB', 'ConvE', 'RandomBaseline',
           'Loss', 'AbsoluteMarginLoss', 'SelfAdversarialLoss', 'NLLLoss', 'PairwiseLoss', 'BCELoss', 'NLLMulticlass',
           'Regularizer', 'LPRegularizer', 'Optimizer', 'AdagradOptimizer', 'AdamOptimizer', 'MomentumOptimizer',
           'SGDOptimizer', 'Initializer', 'RandomNormal', 'RandomUniform', 'Xavier', 'Constant',
           'get_entity_triples',
           'save_model', 'restore_model']
