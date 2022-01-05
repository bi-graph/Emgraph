
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
        Add a Tensorflow node for the initializer.

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

