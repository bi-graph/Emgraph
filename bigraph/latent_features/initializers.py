import tensorflow as tf
import abc
import logging
import numpy as np
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INITIALIZER_REGISTRY = {}

# Default value of lower bound for uniform sampling
DEFAULT_UNIFORM_LOW = -0.05

# Default value of upper bound for uniform sampling
DEFAULT_UNIFORM_HIGH = 0.05

# Default value of mean for Gaussian sampling
DEFAULT_NORMAL_MEAN = 0

# Default value of std for Gaussian sampling
DEFAULT_NORMAL_STD = 0.05

# Default value indicating whether to use xavier uniform or normal
DEFAULT_XAVIER_IS_UNIFORM = False


def register_initializer(name, external_params=[], class_params={}):
    """
    Wrapper for Saving the initializer class info in the INITIALIZER_REGISTRY dictionary.

    :param name: Name of the class
    :type name: str
    :param external_params: External parameters
    :type external_params: list
    :param class_params: Class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """

    def insert_in_registry(class_handle):
        INITIALIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        INITIALIZER_REGISTRY[name].external_params = external_params
        INITIALIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class Initializer(abc.ABC):
    """
    Abstract class for initializers.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """
        Initialize the class.

        :param initializer_params: Dictionary of hyperparams that would be used by the initializer
        :type initializer_params: dict
        :param verbose: Set/reset verbose mode
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int/np.random.RandomState
        """
        self.verbose = verbose
        self._initializer_params = {}
        if isinstance(seed, int):
            self.random_generator = check_random_state(seed)
        else:
            self.random_generator = seed
        self._init_hyperparams(initializer_params)

    def _display_params(self):
        """
        Display the parameter values.

        """
        logger.info('\n------ Initializer -----')
        logger.info('Name : {}'.format(self.name))
        for key, value in self._initializer_params.items():
            logger.info('{} : {}'.format(key, value))

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the parameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        """
        raise NotImplementedError('Abstract Method not implemented!')

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
        raise NotImplementedError('Abstract Method not implemented!')

    def _get_np_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """
        Generate an initialized Numpy array for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initialized weights
        :rtype: nd-array
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def get_entity_initializer(self, in_shape=None, out_shape=None, init_type='tf'):
        """
        Entity embedding initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param init_type: Initializer type (tf: Tensorflow, np: Numpy)
        :type init_type: str
        :return: Weights initializer
        :rtype: tf.Op / nd-array
        """

        assert init_type in ['tf', 'np'], 'Invalid initializer type!'
        if init_type == 'tf':
            return self._get_tf_initializer(in_shape, out_shape, 'e')
        else:
            return self._get_np_initializer(in_shape, out_shape, 'e')

    def get_relation_initializer(self, in_shape=None, out_shape=None, init_type='tf'):
        """
        Relation embeddings' initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's outputs.
        :type out_shape: int
        :param init_type: Initializer type
        :type init_type: str
        :return: Weights initializer
        :rtype: tf.Op / nd-array
        """
        assert init_type in ['tf', 'np'], 'Invalid initializer type!'
        if init_type == 'tf':
            return self._get_tf_initializer(in_shape, out_shape, 'r')
        else:
            return self._get_np_initializer(in_shape, out_shape, 'r')


@register_initializer("normal", ["mean", "std"])
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

        self._initializer_params['mean'] = hyperparam_dict.get('mean', DEFAULT_NORMAL_MEAN)
        self._initializer_params['std'] = hyperparam_dict.get('std', DEFAULT_NORMAL_STD)

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

        return tf.random_normal_initializer(mean=self._initializer_params['mean'],
                                            stddev=self._initializer_params['std'],
                                            dtype=tf.float32)

    def _get_np_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """
        Generate an initialized Numpy array for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initialized weights
        :rtype: nd-array
        """

        return self.random_generator.normal(self._initializer_params['mean'],
                                            self._initializer_params['std'],
                                            size=(in_shape, out_shape)).astype(np.float32)


@register_initializer("uniform", ["low", "high"])
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
        return tf.random_uniform_initializer(minval=self._initializer_params['low'],
                                             maxval=self._initializer_params['high'],
                                             dtype=tf.float32)

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
        return self.random_generator.uniform(self._initializer_params['low'],
                                             self._initializer_params['high'],
                                             size=(in_shape, out_shape)).astype(np.float32)


@register_initializer("xavier", ["uniform"])
class Xavier(Initializer):
    r"""
    Sample from a Xavier distribution :cite:`glorot2010understanding`.
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
            - **uniform**: (float): Xavier Uniform or Xavier Normal initializer
        Example: `initializer_params={'low': 0.9, 'high': 0.02}`
        :type initializer_params: dict
        :param verbose: Activate verbose
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int
        """

        super(Xavier, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value pairs. The initializer gets the params from the keys
        :type hyperparam_dict: dict
        """
        self._initializer_params['uniform'] = hyperparam_dict.get('uniform', DEFAULT_XAVIER_IS_UNIFORM)

        if self.verbose:
            self._display_params()