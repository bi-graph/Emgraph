

import os
import pickle
import importlib
from time import gmtime, strftime
import glob
import logging

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import pandas as pd

"""This module contains utility functions for neural knowledge graph embedding models.
"""

DEFAULT_MODEL_NAMES = "{0}.model.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

