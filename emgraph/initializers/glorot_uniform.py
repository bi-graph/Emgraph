import numpy as np
import tensorflow as tf

from emgraph.initializers._initializer_constants import DEFAULT_GLOROT_IS_UNIFORM
from emgraph.initializers.initializer import Initializer
from emgraph.initializers.utils import export_emgraph_initializer


@export_emgraph_initializer("glorot_uniform", ["uniform"])
class GlorotUniform(Initializer):
    r"""
    Sample from a GlorotUniform distribution :cite:`glorot2010understanding`.
    If uniform is set to True:
        .. math::

            \mathcal{U} ( - \sqrt{ \frac{6}{ fan_{in} + fan_{out} } }, \sqrt{ \frac{6}{ fan_{in} + fan_{out} } } )
    else:
        .. math::

            \mathcal{N} ( 0, \sqrt{ \frac{2}{ fan_{in} + fan_{out} } } )

    where :math:`fan_{in}` and :math:`fan_{out}` are Number of the layer's inputs and outputs respectively.
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """
        Initialize the Uniform class.

        :param initializer_params: Key-value pairs. The initializer gets the params from the keys:
            - **uniform**: (float): GlorotUniform Uniform or GlorotUniform Normal initializer
        Example: `initializer_params={'low': 0.9, 'high': 0.02}`
        :type initializer_params: dict
        :param verbose: Activate verbose
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int
        """

        super(GlorotUniform, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value pairs. The initializer gets the params from the keys
        :type hyperparam_dict: dict
        """
        self._initializer_params['uniform'] = hyperparam_dict.get('uniform', DEFAULT_GLOROT_IS_UNIFORM)

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
        # return tf.contrib.layers.xavier_initializer(uniform=self._initializer_params['uniform'], dtype=tf.float32)
        # return tf.initializers.glorot_uniform()
        return tf.initializers.GlorotUniform()

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

        if self._initializer_params['uniform']:
            limit = np.sqrt(6 / (in_shape + out_shape))
            return self.random_generator.uniform(-limit, limit, size=(in_shape, out_shape)).astype(np.float32)
        else:
            std = np.sqrt(2 / (in_shape + out_shape))
            return self.random_generator.normal(0, std, size=(in_shape, out_shape)).astype(np.float32)
