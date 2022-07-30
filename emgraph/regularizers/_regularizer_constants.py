import logging

REGULARIZER_REGISTRY = {}
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# default lambda to be used in L1, L2 and L3 regularizer
DEFAULT_LAMBDA = 1e-5
# default regularization - L2
DEFAULT_NORM = 2
