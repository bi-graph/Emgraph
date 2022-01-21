
import tensorflow as tf
import abc
import logging

import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OPTIMIZER_REGISTRY = {}

def register_optimizer(name, external_params=[], class_params={}):
    """
    A wrapper for saving the wrapped optimizer in a dictionary.

    :param name: name of the loss-function
    :type name: str
    :param external_params: A list containing the external parameters of the loss-function
    :type external_params: list
    :param class_params: A dictionary containing the class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """
    def insert_in_registry(class_handle):
        OPTIMIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        OPTIMIZER_REGISTRY[name].external_params = external_params
        OPTIMIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry