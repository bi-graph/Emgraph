import abc

import tensorflow as tf

from emgraph.losses._loss_constants import LOSS_REGISTRY, logger


class Loss(abc.ABC):
    """
    Abstract class for the loss functions.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, eta, hyperparam_dict, verbose=False):
        """
        Initialize the Loss class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        self._loss_parameters = {}
        self._dependencies = []

        # perform check to see if all the required external hyperparams are passed
        try:
            self._loss_parameters['eta'] = eta
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                logger.info('\n--------- Loss ---------')
                logger.info('Name : {}'.format(self.name))
                for key, value in self._loss_parameters.items():
                    logger.info('{} : {}'.format(key, value))
        except KeyError as e:
            msg = 'Some of the hyperparams for loss are not passed to the loss function.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return:
        :rtype:
        """

        msg = 'This function is a placeholder in an abstract class'
        logger.error(msg)
        NotImplementedError(msg)

    def get_state(self, param_name):
        """
        Get the state value.

        :param param_name: State name which is querying the value
        :type param_name: str
        :return: The value of the corresponding state
        :rtype: str
        """

        try:
            param_value = LOSS_REGISTRY[self.name].class_params.get(param_name)
            return param_value
        except KeyError as e:
            msg = 'Invalid Key.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def _inputs_check(self, scores_pos, scores_neg):
        """
        Check and create dependencies needed by loss computations.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        """

        logger.debug('Creating dependencies before loss computations.')
        self._dependencies = []
        if LOSS_REGISTRY[self.name].class_params['require_same_size_pos_neg'] and self._loss_parameters['eta'] != 1:
            logger.debug('Dependencies found: \n\tRequired same size positive and negative. \n\tEta is not 1.')
            self._dependencies.append(
                tf.Assert(
                    tf.equal(tf.shape(scores_pos)[0], tf.shape(scores_neg)[0]),
                    [tf.shape(scores_pos)[0], tf.shape(scores_neg)[0]]
                    )
                )

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss-function. All child classes must override this method.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: float
        """

        msg = 'This function is a placeholder in an abstract class.'
        logger.error(msg)
        NotImplementedError(msg)

    def apply(self, scores_pos, scores_neg):
        """
        Interface of the Loss class. Check, preprocess the inputs and apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        self._inputs_check(scores_pos, scores_neg)
        with tf.control_dependencies(self._dependencies):
            loss = self._apply(scores_pos, scores_neg)
        return loss
