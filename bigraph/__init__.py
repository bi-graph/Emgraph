# Copyright (C) 2017-2022 Bigraph developers
# Author: Soran Ghadri
# Contact: soran.gdr.cs@gmail.com

"""Bigraph is a Python toolkit for graph embedding and link prediction."""

import logging.config
import pkg_resources

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_version() -> str:
    __version__ = '1.0rc1'
    return __version__


__all__ = ['preprocessing', 'predict', 'datasets', 'latent_features', 'evaluation', 'utils']

# logging.config.fileConfig(pkg_resources.resource_filename(__name__, 'logger.conf'), disable_existing_loggers=False)
