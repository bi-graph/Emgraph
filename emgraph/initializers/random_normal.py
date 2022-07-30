import numpy as np
import tensorflow as tf

from emgraph.initializers._initializer_constants import (
    DEFAULT_NORMAL_MEAN,
    DEFAULT_NORMAL_STD,
)
from emgraph.initializers.initializer import Initializer
from emgraph.initializers.utils import export_emgraph_initializer


@export_emgraph_initializer("normal", ["mean", "std"])
class RandomNormal(Initializer):
    r"""
    Sample from a normal distribution with provided `mean` and `std`.

        .. math::
            \mathcal{N} (\mu, \sigma)
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """
        Initialize the Random Normal class.

        :param initializer_params: Key-value pairs. The initializer gets the params from the keys:
            - **mean**: (float). Mean of the weights(default: 0)
            - **std**: (float). Std of the weights(default: 0.05)
        Example: `initializer_params={'mean': 0.9, 'std': 0.02}`
        :type initializer_params: dict
        :param verbose: Activate verbose
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int / np.random.RandomState
        """

        super(RandomNormal, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value pairs. The initializer gets the params from the keys:
        :type hyperparam_dict: dict
        """

        self._initializer_params["mean"] = hyperparam_dict.get(
            "mean", DEFAULT_NORMAL_MEAN
        )
        self._initializer_params["std"] = hyperparam_dict.get("std", DEFAULT_NORMAL_STD)

        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept="e"):
        """
        Initializer that generates tensors with a normal distribution.
        Initializers allow you to pre-specify an initialization strategy, encoded in the Initializer object, without
        knowing the shape and dtype of the variable being initialized.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Drawn samples from the parameterized normal distribution.
        :rtype: Initializer
        """

        return tf.random_normal_initializer(
            mean=self._initializer_params["mean"],
            stddev=self._initializer_params["std"],
        )

    def _get_np_initializer(self, in_shape=None, out_shape=None, concept="e"):
        """
        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, initially computed by De Moivre and
        independently by Gauss and Laplace 200 years later [2], is commonly referred to as the bell curve because to
        its distinctive form (see the example below).

        Normal distributions appear often in nature. It explains, for example, the often occurring distribution of
        samples altered by a large number of small, random disturbances, each having its own unique distribution [2].

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Drawn samples from the parameterized normal distribution.
        :rtype: ndarray or scalar
        """

        return self.random_generator.normal(
            self._initializer_params["mean"],
            self._initializer_params["std"],
            size=(in_shape, out_shape),
        ).astype(np.float32)
