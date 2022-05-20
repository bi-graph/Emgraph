import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OPTIMIZER_REGISTRY = {}

# Default learning rate for the optimizers
DEFAULT_LR = 0.0005
# Default momentum for the optimizers
DEFAULT_MOMENTUM = 0.9
DEFAULT_DECAY_CYCLE = 0
DEFAULT_DECAY_CYCLE_MULTIPLE = 1
DEFAULT_LR_DECAY_FACTOR = 2
DEFAULT_END_LR = 1e-8
DEFAULT_SINE = False
