
import tensorflow as tf
import abc
import logging

LOSS_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Default margin used by pairwise and absolute margin loss
DEFAULT_MARGIN = 1

# default sampling temperature used by adversarial loss
DEFAULT_ALPHA_ADVERSARIAL = 0.5

# Default margin used by margin based adversarial loss
DEFAULT_MARGIN_ADVERSARIAL = 3

# Min score below which the values will be clipped before applying exponential
DEFAULT_CLIP_EXP_LOWER = -75.0

# Max score above which the values will be clipped before applying exponential
DEFAULT_CLIP_EXP_UPPER = 75.0

# Default label smoothing for ConvE
DEFAULT_LABEL_SMOOTHING = None

# Default label weighting for ConvE
DEFAULT_LABEL_WEIGHTING = False

def register_loss(name, external_params=[], class_params={}):
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

    default_class_params = {'require_same_size_pos_neg': True}

    def populate_class_params():
        LOSS_REGISTRY[name].class_params = {
            'require_same_size_pos_neg': class_params.get('require_same_size_pos_neg',
                                                          default_class_params['require_same_size_pos_neg'])
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
    return tf.clip_by_value(value,
                            clip_value_min=DEFAULT_CLIP_EXP_LOWER,
                            clip_value_max=DEFAULT_CLIP_EXP_UPPER)

class Loss(abc.ABC):
    """
    Abstract class for loss functions.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, eta, hyperparam_dict, verbose=False):
        """
        Initialize the Loss class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        self._loss_parameters = {}
        self._dependencies = []

        # perform check to see if all the required external hyperparams are passed
        try:
            self._loss_parameters['eta'] = eta
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                logger.info('\n--------- Loss ---------')
                logger.info('Name : {}'.format(self.name))
                for key, value in self._loss_parameters.items():
                    logger.info('{} : {}'.format(key, value))
        except KeyError as e:
            msg = 'Some of the hyperparams for loss are not passed to the loss function.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    def _init_hyperparams(self, hyperparam_dict):
        pass

    def get_state(self, param_name):
        """
        Get the state value.

        :param param_name: State name which is querying the value
        :type param_name: str
        :return: The value of the corresponding state
        :rtype: str
        """

        try:
            param_value = LOSS_REGISTRY[self.name].class_params.get(param_name)
            return param_value
        except KeyError as e:
            msg = 'Invalid Key.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)