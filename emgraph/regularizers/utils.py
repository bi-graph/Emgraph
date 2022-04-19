from emgraph.regularizers._regularizer_constants import REGULARIZER_REGISTRY


def export_emgraph_regularizer(name, external_params=None, class_params=None):
    """
    A wrapper for saving the wrapped regularizers in a dictionary.

    :param name: name of the loss-function
    :type name: str
    :param external_params: A list containing the external parameters of the loss-function
    :type external_params: list
    :param class_params: A dictionary containing the class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """

    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        REGULARIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        REGULARIZER_REGISTRY[name].external_params = external_params
        REGULARIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry