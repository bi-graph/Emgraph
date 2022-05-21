import tensorflow as tf

from emgraph.losses._loss_constants import (
    DEFAULT_CLIP_EXP_LOWER,
    DEFAULT_CLIP_EXP_UPPER,
    LOSS_REGISTRY,
)


def export_emgraph_loss(name, external_params=[], class_params={}):
    """
    A wrapper for saving the wrapped loss-function in a dictionary.

    :param name: name of the loss-function
    :type name: str
    :param external_params: A list containing the external parameters of the loss-function
    :type external_params: list
    :param class_params: A dictionary containing the class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """

    default_class_params = {"require_same_size_pos_neg": True}

    def populate_class_params():
        LOSS_REGISTRY[name].class_params = {
            "require_same_size_pos_neg": class_params.get(
                "require_same_size_pos_neg",
                default_class_params["require_same_size_pos_neg"],
            )
        }

    def insert_in_registry(class_handle):
        LOSS_REGISTRY[name] = class_handle
        class_handle.name = name
        LOSS_REGISTRY[name].external_params = external_params
        populate_class_params()
        return class_handle

    return insert_in_registry


def clip_before_exp(value):
    """
    Clip the value for the stability of exponential.

    """
    return tf.clip_by_value(
        value,
        clip_value_min=DEFAULT_CLIP_EXP_LOWER,
        clip_value_max=DEFAULT_CLIP_EXP_UPPER,
    )
