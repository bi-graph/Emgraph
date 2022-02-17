

import os
import pickle
import importlib
from time import gmtime, strftime
import glob
import logging

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import pandas as pd

"""This module contains utility functions for neural knowledge graph embedding models.
"""

DEFAULT_MODEL_NAMES = "{0}.model.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_model(model, model_name_path=None, protocol=pickle.HIGHEST_PROTOCOL):
    """Save the trained model.

    :param model: A trained neural knowledge graph embedding model,
            the model must be an instance of TransE,
            DistMult, ComplEx, or HolE.
    :type model: EmbeddingModel
    :param model_name_path: The name of the model to be saved.
            If not specified, a default name model
            with current datetime is named
            and saved to the working directory
    :type model_name_path: str
    :param protocol: Pickle portocol
    :type protocol: int
    :return:
    :rtype:

    Examples
        --------
        >>> import numpy as np
        >>> from bigraph.latent_features import ComplEx
        >>> from bigraph.utils import save_model
        >>> model = ComplEx(batches_count=2, seed=555, epochs=20, k=10)
        >>> X = np.array([['a', 'y', 'b'],
        >>>               ['b', 'y', 'a'],
        >>>               ['a', 'y', 'c'],
        >>>               ['c', 'y', 'a'],
        >>>               ['a', 'y', 'd'],
        >>>               ['c', 'y', 'd'],
        >>>               ['b', 'y', 'c'],
        >>>               ['f', 'y', 'e']])
        >>> model.fit(X)
        >>> y_pred_before = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
        >>> example_name = 'helloworld.pkl'
        >>> save_model(model, model_name_path = example_name)
        >>> print(y_pred_before)
        [-0.29721245, 0.07865551]
    """

    logger.debug('Saving model {}.'.format(model.__class__.__name__))

    obj = {
        'class_name': model.__class__.__name__,
        'hyperparams': model.all_params,
        'is_fitted': model.is_fitted,
        'ent_to_idx': model.ent_to_idx,
        'rel_to_idx': model.rel_to_idx,
        'is_calibrated': model.is_calibrated
    }

    model.get_embedding_model_params(obj)

    logger.debug('Saving hyperparams:{}\n\tis_fitted: \
                 {}'.format(model.all_params, model.is_fitted))

    if model_name_path is None:
        model_name_path = DEFAULT_MODEL_NAMES.format(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))

    with open(model_name_path, 'wb') as fw:
        pickle.dump(obj, fw, protocol=protocol)
        # dump model tf
