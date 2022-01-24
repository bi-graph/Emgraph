
import tensorflow as tf
import abc
import logging

import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OPTIMIZER_REGISTRY = {}

# Default learning rate for the optimizers
DEFAULT_LR = 0.0005

# Default momentum for the optimizers
DEFAULT_MOMENTUM = 0.9

DEFAULT_DECAY_CYCLE = 0

DEFAULT_DECAY_CYCLE_MULTIPLE = 1

DEFAULT_LR_DECAY_FACTOR = 2

DEFAULT_END_LR = 1e-8

DEFAULT_SINE = False

def register_optimizer(name, external_params=[], class_params={}):
    """
    A wrapper for saving the wrapped optimizer in a dictionary.

    :param name: name of the loss-function
    :type name: str
    :param external_params: A list containing the external parameters of the loss-function
    :type external_params: list
    :param class_params: A dictionary containing the class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """
    def insert_in_registry(class_handle):
        OPTIMIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        OPTIMIZER_REGISTRY[name].external_params = external_params
        OPTIMIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry

class Optimizer(abc.ABC):
    """
    Abstract class for the optimizers.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, optimizer_params, batches_count, verbose):
        """
        Initialize the optimizer.

        :param optimizer_params: Key-value dictionary for parameters.
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        self.verbose = verbose
        self._optimizer_params = {}
        self._init_hyperparams(optimizer_params)
        self.batches_count = batches_count


    def _display_params(self):
        """
        Display the parameter values.

        """
        logger.info('\n------ Optimizer -----')
        logger.info('Name : {}'.format(self.name))
        for key, value in self._optimizer_params.items():
            logger.info('{} : {}'.format(key, value))

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._optimizer_params['lr'] = hyperparam_dict.get('lr', DEFAULT_LR)
        if self.verbose:
            self._display_params()

    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """
        Update values of placeholders created by the optimizer.

        :param feed_dict: model sess.run feeding dictionary while being optimized
        :type feed_dict: dict
        :param batch_num: Current batch number
        :type batch_num: int
        :param epoch_num: Current epoch number
        :type epoch_num: int
        :return: -
        :rtype: -
        """

        raise NotImplementedError('Abstract Method not implemented!')


@register_optimizer("adagrad", ['lr'])
class AdagradOptimizer(Optimizer):
    """
    Wrapper around adagrad optimizer.

    """

    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize  the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate (default: 0.0005)
            Example: ``optimizer_params={'lr': 0.001}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(AdagradOptimizer, self).__init__(optimizer_params, batches_count, verbose)