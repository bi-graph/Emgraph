import abc

from emgraph.training._optimizer_constants import DEFAULT_LR, logger


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

    def minimize(self, loss, var_list):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :param var_list: A list of variables to train
        :type var_list: list
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
