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

