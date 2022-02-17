

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

    Examples:

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
        >>> example_name = 'bigraph_model.pkl'
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


def restore_model(model_name_path=None):
    """Restore a saved model from disk.

        See also :meth:`save_model`.

    :param model_name_path: The name of saved model to be restored. If not specified,
    the library will try to find the default model in the working directory.
    :type model_name_path: str
    :return: The neural knowledge graph embedding model restored from disk.
    :rtype: EmbeddingModel

    Examples:

        >>> from bigraph.utils import restore_model
        >>> import numpy as np
        >>> example_name = 'bigraph_model.pkl'
        >>> restored_model = restore_model(model_name_path = example_name)
        >>> y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
        >>> print(y_pred_after)
        [-0.29721245, 0.07865551]
    """

    if model_name_path is None:
        logger.warning("There is no model name specified. \
                        We will try to lookup \
                        the latest default saved model...")
        default_models = glob.glob("*.model.pkl")
        if len(default_models) == 0:
            raise Exception("No default model found. Please specify \
                             model_name_path...")
        else:
            model_name_path = default_models[len(default_models) - 1]
            logger.info("Will will load the model: {0} in your \
                         current dir...".format(model_name_path))

    model = None
    logger.info('Will load model {}.'.format(model_name_path))

    try:
        with open(model_name_path, 'rb') as fr:
            restored_obj = pickle.load(fr)

        logger.debug('Restoring model ...')
        module = importlib.import_module("ampligraph.latent_features")
        class_ = getattr(module, restored_obj['class_name'])
        model = class_(**restored_obj['hyperparams'])
        model.is_fitted = restored_obj['is_fitted']
        model.ent_to_idx = restored_obj['ent_to_idx']
        model.rel_to_idx = restored_obj['rel_to_idx']

        try:
            model.is_calibrated = restored_obj['is_calibrated']
        except KeyError:
            model.is_calibrated = False

        model.restore_model_params(restored_obj)
    except pickle.UnpicklingError as e:
        msg = 'Error unpickling model {} : {}.'.format(model_name_path, e)
        logger.debug(msg)
        raise Exception(msg)
    except (IOError, FileNotFoundError):
        msg = 'No model found: {}.'.format(model_name_path)
        logger.debug(msg)
        raise FileNotFoundError(msg)

    return model