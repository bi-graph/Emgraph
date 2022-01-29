
import tensorflow as tf
import numpy as np
import abc
import logging

REGULARIZER_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# defalut lambda to be used in L1, L2 and L3 regularizer
DEFAULT_LAMBDA = 1e-5

# default regularization - L2
DEFAULT_NORM = 2


def register_regularizer(name, external_params=None, class_params=None):
    """
    A wrapper for saving the wrapped regularizers in a dictionary.

    :param name: name of the loss-function
    :type name: str
    :param external_params: A list containing the external parameters of the loss-function
    :type external_params: list
    :param class_params: A dictionary containing the class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """

    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        REGULARIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        REGULARIZER_REGISTRY[name].external_params = external_params
        REGULARIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class Regularizer(abc.ABC):
    """
    Abstract class for Regularizer.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, hyperparam_dict, verbose=False):
        """

        :param hyperparam_dict: Key-value dictionary for parameters.
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        self._regularizer_parameters = {}

        # perform check to see if all the required external hyperparams are passed
        try:
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                logger.info('\n------ Regularizer -----')
                logger.info('Name : {}'.format(self.name))
                for key, value in self._regularizer_parameters.items():
                    logger.info('{} : {}'.format(key, value))

        except KeyError as e:
            msg = 'Some of the hyperparams for regularizer were not passed.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)


    def get_state(self, param_name):
        """
        Get the state.

        :param param_name: Name of the state being queried
        :type param_name: str
        :return: Value of the queried state
        :rtype:
        """

        try:
            param_value = REGULARIZER_REGISTRY[self.name].class_params.get(param_name)
            return param_value
        except KeyError as e:
            msg = 'Invalid Key.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        logger.error('This function is a placeholder in an abstract class')
        raise NotImplementedError("This function is a placeholder in an abstract class")

    def _apply(self, trainable_params):
        """
        Apply the regularization function. All children must override this method.

        :param trainable_params: Trainable to-be-regularized params' list
        :type trainable_params: list, shape [n]
        :return: Regularization loss
        :rtype: tf.Tensor
        """

        logger.error('This function is a placeholder in an abstract class')
        raise NotImplementedError("This function is a placeholder in an abstract class")

    def apply(self, trainable_params):
        """
        Interface of the Loss class. Apply the regularization function. All children must override this method.

        :param trainable_params: Trainable to-be-regularized params' list
        :type trainable_params: list, shape [n]
        :return: Regularization loss
        :rtype: tf.Tensor
        """
        loss = self._apply(trainable_params)
        return loss