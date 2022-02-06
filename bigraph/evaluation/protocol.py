from collections.abc import Iterable
from itertools import product, islice
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf


from ..evaluation import mrr_score, hits_at_n_score, mr_score
from ..datasets import BigraphDatasetAdapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOO_MANY_ENTITIES_TH = 50000


def _create_unique_mappings(unique_obj, unique_rel):
    """Create unique mappings.

    :param unique_obj: Unique object
    :type unique_obj: list
    :param unique_rel: Unique relationship
    :type unique_rel: list
    :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping: entity to idx mapping
    :rtype: dict, dict
    """

    obj_count = len(unique_obj)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    obj_to_idx = dict(zip(unique_obj, range(obj_count)))
    return rel_to_idx, obj_to_idx


def create_mappings(X):
    """Create string-IDs mappings for entities and relations.

    Entities and relations are assigned incremental, unique integer IDs.
    Mappings are preserved in two distinct dictionaries,
    and counters are separated for entities and relations mappings.

    :param X: The triples to extract mappings
    :type X: ndarray, shape [n, 3]
    :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping: entity to idx mapping
    :rtype: dict, dict
    """

    logger.debug('Creating mappings for entities and relations.')
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
    unique_rel = np.unique(X[:, 1])
    return _create_unique_mappings(unique_ent, unique_rel)

def _convert_to_idx(X, ent_to_idx, rel_to_idx, obj_to_idx):
    """Convert statements (triples) into integer IDs.

    :param X: The statements to be converted.
    :type X: ndarray
    :param ent_to_idx: Entity to idx mappings
    :type ent_to_idx: dict
    :param rel_to_idx: Relation to idx mappings
    :type rel_to_idx: dict
    :param obj_to_idx: Object to idx mappings
    :type obj_to_idx: dict
    :return: Converted statements
    :rtype: ndarray, shape [n, 3]
    """
    unseen_msg = 'Input triples include one or more {concept_type} not present in the training set. ' \
                 'Please filter all concepts in X that do not occur in the training test ' \
                 '(set filter_unseen=True in evaluate_performance) or retrain the model on a ' \
                 'training set that includes all the desired concept types.'

    try:
        x_idx_s = np.vectorize(ent_to_idx.get)(X[:, 0])
        x_idx_p = np.vectorize(rel_to_idx.get)(X[:, 1])
        x_idx_o = np.vectorize(obj_to_idx.get)(X[:, 2])
    except TypeError:
        unseen_msg = unseen_msg.format(**{'concept_type': 'concepts'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_s or None in x_idx_o:
        unseen_msg = unseen_msg.format(**{'concept_type': 'entities'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_p:
        unseen_msg = unseen_msg.format(**{'concept_type': 'relations'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    return np.dstack([x_idx_s, x_idx_p, x_idx_o]).reshape((-1, 3))


def to_idx(X, ent_to_idx, rel_to_idx):
    """Convert statements (triples) into integer IDs.

    :param X: The statements to be converted.
    :type X: ndarray
    :param ent_to_idx: Entity to idx mappings
    :type ent_to_idx: dict
    :param rel_to_idx: Relation to idx mappings
    :type rel_to_idx: dict
    :return: Converted statements
    :rtype: ndarray, shape [n, 3]
    """

    logger.debug('Converting statements to integer ids.')
    if X.ndim == 1:
        X = X[np.newaxis, :]
    return _convert_to_idx(X, ent_to_idx, rel_to_idx, ent_to_idx)