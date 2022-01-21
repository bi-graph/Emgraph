
import tensorflow as tf
import abc
import logging

import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OPTIMIZER_REGISTRY = {}

