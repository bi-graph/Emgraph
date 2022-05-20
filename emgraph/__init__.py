# Copyright (C) 2019-2022 Emgraph developers
# Author: Soran Ghaderi
# Contact: soran.gdr.cs@gmail.com

"""Emgraph is a Python toolkit for graph embedding and link prediction."""

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from . import datasets, evaluation, initializers, layers, losses, models, regularizers, training


def get_version() -> str:
    __version__ = '1.0rc2'
    return __version__


__all__ = ['datasets', 'evaluation', 'initializers', 'layers', 'losses', 'models', 'regularizers', 'training']
