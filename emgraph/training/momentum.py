import tensorflow as tf

from emgraph.training._optimizer_constants import DEFAULT_LR, DEFAULT_MOMENTUM
from emgraph.training.optimizer import Optimizer
from emgraph.training.utils import export_emgraph_optimizer


@export_emgraph_optimizer(name="momentum", external_params=["lr", "momentum"])
class Momentum(Optimizer):
    """
    Momentum Optimizer.

    """

    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate (default: 0.0005)
            - **'momentum'**: (float). Momentum (default: 0.9)\

            Example: ``optimizer_params={'lr': 0.001, 'momentum':0.90}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(Momentum, self).__init__(optimizer_params, batches_count, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._optimizer_params["lr"] = hyperparam_dict.get("lr", DEFAULT_LR)
        self._optimizer_params["momentum"] = hyperparam_dict.get(
            "momentum", DEFAULT_MOMENTUM
        )

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

        self.optimizer = tf.optimizers.SGD(
            learning_rate=self._optimizer_params["lr"],
            momentum=self._optimizer_params["momentum"],
        )

        train = self.optimizer.minimize(loss, var_list)
        return train

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
        return
