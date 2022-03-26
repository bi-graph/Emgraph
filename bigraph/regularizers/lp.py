import numpy as np
import tensorflow as tf

from bigraph.regularizers._regularizer_constants import logger, DEFAULT_LAMBDA, DEFAULT_NORM
from bigraph.regularizers.utils import export_emgraph_regularizer
from bigraph.regularizers.regularizer import Regularizer


@export_emgraph_regularizer("LP", ['p', 'lambda'])
class LPRegularizer(Regularizer):
    r"""
    LP regularizer class

        .. math::

               \mathcal{L}(Reg) =  \sum_{i=1}^{n}  \lambda_i * \mid w_i \mid_p

    where n is the number of model parameters, :math:`p \in{1,2,3}` is the p-norm and
    :math:`\lambda` is the regularization weight.

    For example, if :math:`p=1` the function will perform L1 regularization.
    L2 regularization is obtained with :math:`p=2`.

    The nuclear 3-norm proposed in the ComplEx-N3 paper :cite:`lacroix2018canonical` can be obtained with
    ``regularizer_params={'p': 3}``.
    """

    def __init__(self, regularizer_params=None, verbose=False):
        """
        Initialize the regularizer hyperparameters.

        :param regularizer_params: Key-value dictionary for hyperparameters.

            - **'lambda'**: (float). Weight of regularization loss for each parameter (default: 1e-5)
            - **'p'**: (int): norm (default: 2)

            Example: ``regularizer_params={'lambda': 1e-5, 'p': 1}``

        :type regularizer_params: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        if regularizer_params is None:
            regularizer_params = {'lambda': DEFAULT_LAMBDA, 'p': DEFAULT_NORM}
        super().__init__(regularizer_params, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.

        lambda': list or float
                weight for regularizer loss for each parameter(default: 1e-5).
                If list, size must be equal to no. of parameters.

            'p': int
                Norm of the regularizer (``1`` for L1 regularizer, ``2`` for L2 and so on.) (default:2)

        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._regularizer_parameters['lambda'] = hyperparam_dict.get('lambda', DEFAULT_LAMBDA)
        self._regularizer_parameters['p'] = hyperparam_dict.get('p', DEFAULT_NORM)
        if not isinstance(self._regularizer_parameters['p'], (int, np.integer)):
            msg = 'Invalid value for regularizer parameter p:{}. Supported type int, np.int32 or np.int64'.format(
                self._regularizer_parameters['p'])
            logger.error(msg)
            raise Exception(msg)

    def _apply(self, trainable_params):
        """
        Apply the regularizer.

        :param trainable_params: Trainable parameters that are going to be regularized
        :type trainable_params: list, shape [n]
        :return: Regularization loss
        :rtype: tf.Tensor
        """

        if np.isscalar(self._regularizer_parameters['lambda']):
            self._regularizer_parameters['lambda'] = [self._regularizer_parameters['lambda']] * len(trainable_params)
        elif isinstance(self._regularizer_parameters['lambda'], list) and len(
                self._regularizer_parameters['lambda']) == len(trainable_params):
            pass
        else:
            logger.error('Regularizer weight must be a scalar or a list with length equal to number of params passes')
            raise ValueError(
                "Regularizer weight must be a scalar or a list with length equal to number of params passes")

        loss_reg = 0
        for i in range(len(trainable_params)):
            loss_reg += (self._regularizer_parameters['lambda'][i] * tf.reduce_sum(
                tf.pow(tf.abs(trainable_params[i]), self._regularizer_parameters['p'])))

        return loss_reg