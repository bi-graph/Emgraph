import numpy as np
import tensorflow as tf

from emgraph.initializers._initializer_constants import DEFAULT_UNIFORM_HIGH, DEFAULT_UNIFORM_LOW
from emgraph.initializers.initializer import Initializer
from emgraph.initializers.utils import export_emgraph_initializer


@export_emgraph_initializer("uniform", ["low", "high"])
class RandomUniform(Initializer):
    r"""
    Sample from a normal distribution with provided `mean` and `std`.

        .. math::
            \mathcal{U} (low, high)
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """
        Initialize the Uniform class.

        :param initializer_params: Key-value pairs. The initializer gets the params from the keys:
            - **low**: (float). lower bound for uniform number (default: -0.05)
            - **high**: (float): upper bound for uniform number (default: 0.05)
        Example: `initializer_params={'low': 0.9, 'high': 0.02}`
        :type initializer_params: dict
        :param verbose: Activate verbose
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int
        """

        super(RandomUniform, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value pairs. The initializer gets the params from the keys
        :type hyperparam_dict: dict
        """

        self._initializer_params['low'] = hyperparam_dict.get('low', DEFAULT_UNIFORM_LOW)
        self._initializer_params['high'] = hyperparam_dict.get('high', DEFAULT_UNIFORM_HIGH)

        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """
        Generate an initialized Tensorflow node for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initializer instance
        :rtype: Initializer
        """

        return tf.random_uniform_initializer(
            minval=self._initializer_params['low'],
            maxval=self._initializer_params['high'],
            dtype=tf.float32
        )

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """
        Generate an initialized Numpy array for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initialized weights (uniform distribution)
        :rtype: nd-array
        """

        return self.random_generator.uniform(
            self._initializer_params['low'],
            self._initializer_params['high'],
            size=(in_shape, out_shape)
        ).astype(np.float32)
