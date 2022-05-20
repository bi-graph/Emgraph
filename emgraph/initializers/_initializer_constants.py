import logging

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
# Default value indicating whether to use glorot uniform or normal
DEFAULT_GLOROT_IS_UNIFORM = False
