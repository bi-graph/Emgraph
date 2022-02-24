
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
import abc
from tqdm import tqdm
import logging
from bigraph.latent_features.loss_functions import LOSS_REGISTRY
from bigraph.latent_features.regularizers import REGULARIZER_REGISTRY
from bigraph.latent_features.optimizers import OPTIMIZER_REGISTRY, SGDOptimizer
from bigraph.latent_features.initializers import INITIALIZER_REGISTRY, DEFAULT_XAVIER_IS_UNIFORM
from bigraph.evaluation import generate_corruptions_for_fit, to_idx, generate_corruptions_for_eval, \
    hits_at_n_score, mrr_score
from bigraph.datasets import BigraphDatasetAdapter, NumpyDatasetAdapter
from functools import partial
from bigraph.latent_features import constants as constants
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_REGISTRY = {}

ENTITY_THRESHOLD = 5e5


def set_entity_threshold(threshold):
    """Sets the entity threshold (threshold after which large graph mode is initiated)

    :param threshold: Threshold for a graph to be considered as a big graph
    :type threshold: int
    :return:
    :rtype:
    """

    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = threshold



def reset_entity_threshold():
    """Resets the entity threshold
    """
    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = 5e5


def register_model(name, external_params=None, class_params=None):
    """Wrapper for Saving the class info in the MODEL_REGISTRY dictionary.

    :param name: Name of the class
    :type name: str
    :param external_params: External parameters
    :type external_params: list
    :param class_params: Class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        MODEL_REGISTRY[name] = class_handle
        class_handle.name = name
        MODEL_REGISTRY[name].external_params = external_params
        MODEL_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry