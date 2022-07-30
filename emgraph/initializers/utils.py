from emgraph.initializers._initializer_constants import INITIALIZER_REGISTRY


def export_emgraph_initializer(name, external_params=[], class_params={}):
    """
    Wrapper for Saving the initializer class info in the INITIALIZER_REGISTRY dictionary.

    :param name: Name of the class
    :type name: str
    :param external_params: External parameters
    :type external_params: list
    :param class_params: Class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """

    def insert_in_registry(class_handle):
        INITIALIZER_REGISTRY[name] = class_handle
        class_handle.name = name
        INITIALIZER_REGISTRY[name].external_params = external_params
        INITIALIZER_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry
