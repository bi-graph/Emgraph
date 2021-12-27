
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

        self.verbose = verbose
        self._initializer_params = {}
        if isinstance(seed, int):
            self.random_generator = check_random_state(seed)
        else:
            self.random_generator = seed
        self._init_hyperparams(initializer_params)


    def _display_params(self):

        logger.info('\n------ Initializer -----')
        logger.info('Name : {}'.format(self.name))
        for key, value in self._initializer_params.items():
            logger.info('{} : {}'.format(key, value))