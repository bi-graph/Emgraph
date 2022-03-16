

import numpy as np
import tensorflow as tf
import logging
from sklearn.utils import check_random_state
from tqdm import tqdm
from functools import partial
import time

from .EmbeddingModel import EmbeddingModel, register_model, ENTITY_THRESHOLD
from ..initializers import DEFAULT_XAVIER_IS_UNIFORM
from bigraph.latent_features import constants as constants

from ...datasets import OneToNDatasetAdapter
from ..optimizers import SGDOptimizer
from ...evaluation import to_idx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)