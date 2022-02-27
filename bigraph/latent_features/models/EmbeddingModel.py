# todo: rename most of these functions' names
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

# todo: rename this
@tf.custom_gradient
def custom_softplus(x):
    e = 9999 * tf.exp(x)

    def grad(dy):
        return dy * (1 - 1 / (1 + e))

    return tf.math.log(1 + e), grad


class EmbeddingModel(abc.ABC):
    """Abstract class for embedding models

    BiGraph neural knowledge graph embeddings models extend this class and
    its core methods.

    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 large_graphs=False,
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize the EmbeddingModel class

        Also creates a new Tensorflow session for training.

        :param k: Embedding space dimensionality.
        :type k: int
        :param eta: The number of negatives that must be generated at runtime during training for each positive.
        :type eta: int
        :param epochs: The iterations of the training loop.
        :type epochs: int
        :param batches_count: The number of batches in which the training set must be split during the training loop.
        :type batches_count: int
        :param seed: The seed used by the internal random numbers generator.
        :type seed: int
        :param embedding_model_params: Model-specific hyperparams, passed to the model as a dictionary.
        Refer to model-specific documentation for details.

        For FocusE Layer, following hyper-params can be passed:

            - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
            - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value
            until it reaches original value.
            - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph
            topology.
        - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]
        :type embedding_model_params: dict
        :param optimizer: The optimizer used to minimize the loss function. Choose between
            'sgd', 'adagrad', 'adam', 'momentum'.
        :type optimizer: str
        :param optimizer_params: Arguments specific to the optimizer, passed as a dictionary.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.009}``
        :type optimizer_params: dict
        :param loss: The type of loss function to use during training.

            - ``pairwise``  the model will use pairwise margin-based loss function.
            - ``nll`` the model will use negative loss likelihood.
            - ``absolute_margin`` the model will use absolute margin likelihood.
            - ``self_adversarial`` the model will use adversarial sampling loss function.
            - ``multiclass_nll`` the model will use multiclass nll loss. Switch to multiclass loss defined in
              :cite:`chen2015` by passing 'corrupt_side' as ['s','o'] to embedding_model_params.
              To use loss defined in :cite:`kadlecBK17` pass 'corrupt_side' as 'o' to embedding_model_params.
        :type loss: str
        :param loss_params: Dictionary of loss-specific hyperparameters. See :ref:`loss
            functions <loss>`
            documentation for additional details.

            Example: ``optimizer_params={'lr': 0.01}`` if ``loss='pairwise'``.
        :type loss_params: dict
        :param regularizer: The regularization strategy to use with the loss function.

            - ``None``: the model will not use any regularizer (default)
            - ``LP``: the model will use L1, L2 or L3 based on the value of ``regularizer_params['p']`` (see below).
        :type regularizer: str
        :param regularizer_params: Dictionary of regularizer-specific hyperparameters. See the
            :ref:`regularizers <ref-reg>`
            documentation for additional details.

            Example: ``regularizer_params={'lambda': 1e-5, 'p': 2}`` if ``regularizer='LP'``.
        :type regularizer_params: dict
        :param initializer: The type of initializer to use.

            - ``normal``: The embeddings will be initialized from a normal distribution
            - ``uniform``: The embeddings will be initialized from a uniform distribution
            - ``xavier``: The embeddings will be initialized using xavier strategy (default)
        :type initializer: str
        :param initializer_params: Dictionary of initializer-specific hyperparameters. See the
            :ref:`initializer <ref-init>`
            documentation for additional details.

            Example: ``initializer_params={'mean': 0, 'std': 0.001}`` if ``initializer='normal'``.
        :type initializer_params: dict
        :param large_graphs: Avoid loading entire dataset onto GPU when dealing with large graphs.
        :type large_graphs: bool
        :param verbose: Verbose mode.
        :type verbose: bool
        """

        if (loss == "bce") ^ (self.name == "ConvE"):
            raise ValueError('Invalid Model - Loss combination. '
                             'ConvE model can be used with BCE loss only and vice versa.')

        # Store for restoring later.
        self.all_params = \
            {
                'k': k,
                'eta': eta,
                'epochs': epochs,
                'batches_count': batches_count,
                'seed': seed,
                'embedding_model_params': embedding_model_params,
                'optimizer': optimizer,
                'optimizer_params': optimizer_params,
                'loss': loss,
                'loss_params': loss_params,
                'regularizer': regularizer,
                'regularizer_params': regularizer_params,
                'initializer': initializer,
                'initializer_params': initializer_params,
                'verbose': verbose
            }
        tf.reset_default_graph()
        self.seed = seed
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(seed)

        self.is_filtered = False
        self.use_focusE = False
        self.loss_params = loss_params

        self.embedding_model_params = embedding_model_params

        self.k = k
        self.internal_k = k
        self.epochs = epochs
        self.eta = eta
        self.regularizer_params = regularizer_params
        self.batches_count = batches_count

        self.dealing_with_large_graphs = large_graphs

        if batches_count == 1:
            logger.warning(
                'All triples will be processed in the same batch (batches_count=1). '
                'When processing large graphs it is recommended to batch the input knowledge graph instead.')

        try:
            self.loss = LOSS_REGISTRY[loss](self.eta, self.loss_params, verbose=verbose)
        except KeyError:
            msg = 'Unsupported loss function: {}'.format(loss)
            logger.error(msg)
            raise ValueError(msg)

        try:
            if regularizer is not None:
                self.regularizer = REGULARIZER_REGISTRY[regularizer](self.regularizer_params, verbose=verbose)
            else:
                self.regularizer = regularizer
        except KeyError:
            msg = 'Unsupported regularizer: {}'.format(regularizer)
            logger.error(msg)
            raise ValueError(msg)

        self.optimizer_params = optimizer_params

        try:
            self.optimizer = OPTIMIZER_REGISTRY[optimizer](self.optimizer_params,
                                                           self.batches_count,
                                                           verbose)
        except KeyError:
            msg = 'Unsupported optimizer: {}'.format(optimizer)
            logger.error(msg)
            raise ValueError(msg)

        self.verbose = verbose

        self.initializer_params = initializer_params

        try:
            self.initializer = INITIALIZER_REGISTRY[initializer](self.initializer_params,
                                                                 verbose,
                                                                 self.rnd)
        except KeyError:
            msg = 'Unsupported initializer: {}'.format(initializer)
            logger.error(msg)
            raise ValueError(msg)

        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.trained_model_params = []
        self.is_fitted = False
        self.eval_config = {}
        self.eval_dataset_handle = None
        self.train_dataset_handle = None
        self.is_calibrated = False
        self.calibration_parameters = []

    @abc.abstractmethod
    def _fn(self, e_s, e_p, e_o):
        """The scoring function of the model.

        Assigns a score to a list of triples, with a model-specific strategy.
        Triples are passed as lists of subject, predicate, object embeddings.
        This function must be overridden by every model to return corresponding score.

        :param e_s: The embeddings of a list of subjects.
        :type e_s: Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: Tensor, shape [n]
        :return: The operation corresponding to the scoring function.
        :rtype: tf.Op
        """

        logger.error('_fn is a placeholder function in an abstract class')
        NotImplementedError("This function is a placeholder in an abstract class")


    def get_hyperparameter_dict(self):
        """Return the hyperparameters of the model.

        :return: Dictionary of hyperparameters that were used for training.
        :rtype: dict
        """

        return self.all_params

    def get_embedding_model_params(self, output_dict):
        """Save the model parameters in the dictionary.

        :param output_dict: Saved parameters dictionary. The model saves the parameters, and it can be restored later
        :type output_dict: dict
        :return:
        :rtype:
        """

        output_dict['model_params'] = self.trained_model_params
        output_dict['large_graph'] = self.dealing_with_large_graphs
        output_dict['calibration_parameters'] = self.calibration_parameters

    def restore_model_params(self, in_dict):
        """Load the model parameters from the input dictionary.

        :param in_dict: Saved parameters dictionary. The model loads the parameters.
        :type in_dict: dict
        :return:
        :rtype:
        """

        self.trained_model_params = in_dict['model_params']

        # Try catch is for backward compatibility
        try:
            self.calibration_parameters = in_dict['calibration_parameters']
        except KeyError:
            # For backward compatibility
            self.calibration_parameters = []

        # Try catch is for backward compatibility
        try:
            self.dealing_with_large_graphs = in_dict['large_graph']
        except KeyError:
            # For backward compatibility
            self.dealing_with_large_graphs = False