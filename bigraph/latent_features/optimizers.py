
import tensorflow as tf
import abc
import logging

import math

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

class Optimizer(abc.ABC):
    """
    Abstract class for the optimizers.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, optimizer_params, batches_count, verbose):
        """
        Initialize the optimizer.

        :param optimizer_params: Key-value dictionary for parameters.
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        self.verbose = verbose
        self._optimizer_params = {}
        self._init_hyperparams(optimizer_params)
        self.batches_count = batches_count


    def _display_params(self):
        """
        Display the parameter values.

        """
        logger.info('\n------ Optimizer -----')
        logger.info('Name : {}'.format(self.name))
        for key, value in self._optimizer_params.items():
            logger.info('{} : {}'.format(key, value))

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._optimizer_params['lr'] = hyperparam_dict.get('lr', DEFAULT_LR)
        if self.verbose:
            self._display_params()

    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """
        Update values of placeholders created by the optimizer.

        :param feed_dict: model sess.run feeding dictionary while being optimized
        :type feed_dict: dict
        :param batch_num: Current batch number
        :type batch_num: int
        :param epoch_num: Current epoch number
        :type epoch_num: int
        :return: -
        :rtype: -
        """

        raise NotImplementedError('Abstract Method not implemented!')


@register_optimizer("adagrad", ['lr'])
class AdagradOptimizer(Optimizer):
    """
    Wrapper around adagrad optimizer.

    """

    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize  the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate (default: 0.0005)
            Example: ``optimizer_params={'lr': 0.001}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(AdagradOptimizer, self).__init__(optimizer_params, batches_count, verbose)

    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=self._optimizer_params['lr'])
        train = self.optimizer.minimize(loss)
        return train

    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """
        Update values of placeholders created by the optimizer.

        :param feed_dict: model sess.run feeding dictionary while being optimized
        :type feed_dict: dict
        :param batch_num: Current batch number
        :type batch_num: int
        :param epoch_num: Current epoch number
        :type epoch_num: int
        :return: -
        :rtype: -
        """
        return


@register_optimizer("adam", ['lr'])
class AdamOptimizer(Optimizer):
    """
    Wrapper around Adam Optimizer.

    """

    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate (default: 0.0005)

            Example: ``optimizer_params={'lr': 0.001}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(AdamOptimizer, self).__init__(optimizer_params, batches_count, verbose)

    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._optimizer_params['lr'])

        train = self.optimizer.minimize(loss)
        return train

    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """
        Update values of placeholders created by the optimizer.

        :param feed_dict: model sess.run feeding dictionary while being optimized
        :type feed_dict: dict
        :param batch_num: Current batch number
        :type batch_num: int
        :param epoch_num: Current epoch number
        :type epoch_num: int
        :return: -
        :rtype: -
        """
        return

@register_optimizer("momentum", ['lr', 'momentum'])
class MomentumOptimizer(Optimizer):
    """
    Wrapper around Momentum Optimizer.

    """

    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate (default: 0.0005)
            - **'momentum'**: (float). Momentum (default: 0.9)\

            Example: ``optimizer_params={'lr': 0.001, 'momentum':0.90}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(MomentumOptimizer, self).__init__(optimizer_params, batches_count, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._optimizer_params['lr'] = hyperparam_dict.get('lr', DEFAULT_LR)
        self._optimizer_params['momentum'] = hyperparam_dict.get('momentum', DEFAULT_MOMENTUM)

        if self.verbose:
            self._display_params()


    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self._optimizer_params['lr'],
                                                    momentum=self._optimizer_params['momentum'])

        train = self.optimizer.minimize(loss)
        return train

    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """
        Update values of placeholders created by the optimizer.

        :param feed_dict: model sess.run feeding dictionary while being optimized
        :type feed_dict: dict
        :param batch_num: Current batch number
        :type batch_num: int
        :param epoch_num: Current epoch number
        :type epoch_num: int
        :return: -
        :rtype: -
        """
        return

@register_optimizer("sgd", ['lr', 'decay_cycle', 'end_lr', 'sine_decay', 'expand_factor', 'decay_lr_rate'])
class SGDOptimizer(Optimizer):
    '''
    Wrapper around SGD Optimizer.

    '''
    def __init__(self, optimizer_params, batches_count, verbose=False):
        """
        Initialize  the optimizer.

        :param optimizer_params: Key-value dictionary for hyperparameters.
            - **'lr'**: (float). Learning Rate upper bound (default: 0.0005)
            - **'decay_cycle'**: (int). Cycle of epoch over which to decay (default: 0)
            - **'end_lr'**: (float). Learning Rate lower bound (default: 1e-8)
            - **'cosine_decay'**: (bool). Use cosine decay or to fixed rate decay (default: False)
            - **'expand_factor'**: (float). Expand the decay cycle length by this factor after each cycle \
                (default: 1)
            - **'decay_lr_rate'**: (float). Decay factor to decay the start lr after each cycle \
                (default: 2)

            Example: ``optimizer_params={'lr': 0.01, 'decay_cycle':30, 'end_lr':0.0001, 'sine_decay':True}``
        :type optimizer_params: dict
        :param batches_count: Number of batches per epoch
        :type batches_count: int
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super(SGDOptimizer, self).__init__(optimizer_params, batches_count, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._optimizer_params['lr'] = hyperparam_dict.get('lr', DEFAULT_LR)
        self._optimizer_params['decay_cycle'] = hyperparam_dict.get('decay_cycle', DEFAULT_DECAY_CYCLE)
        self._optimizer_params['cosine_decay'] = hyperparam_dict.get('cosine_decay', DEFAULT_SINE)
        self._optimizer_params['expand_factor'] = hyperparam_dict.get('expand_factor', DEFAULT_DECAY_CYCLE_MULTIPLE)
        self._optimizer_params['decay_lr_rate'] = hyperparam_dict.get('decay_lr_rate', DEFAULT_LR_DECAY_FACTOR)
        self._optimizer_params['end_lr'] = hyperparam_dict.get('end_lr', DEFAULT_END_LR)

        if self.verbose:
            self._display_params()


    def minimize(self, loss):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """

        # create a placeholder for learning rate
        self.lr_placeholder = tf.placeholder(tf.float32)
        # create the optimizer with the placeholder
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_placeholder)

        # load the hyperparameters that would be used while generating the learning rate per batch
        # start learning rate
        self.start_lr = self._optimizer_params['lr']
        self.current_lr = self.start_lr

        # cycle rate for learning rate decay
        self.decay_cycle_rate = self._optimizer_params['decay_cycle']
        self.end_lr = self._optimizer_params['end_lr']

        # check if it is a sinudoidal decay or constant decay
        self.is_cosine_decay = self._optimizer_params['cosine_decay']
        self.next_cycle_epoch = self.decay_cycle_rate + 1

        # Get the cycle expand factor
        self.decay_cycle_expand_factor = self._optimizer_params['expand_factor']

        # Get the LR decay factor at the start of each cycle
        self.decay_lr_rate = self._optimizer_params['decay_lr_rate']
        self.curr_cycle_length = self.decay_cycle_rate
        self.curr_start = 0

        # create the operation that minimizes the loss
        train = self.optimizer.minimize(loss)
        return train