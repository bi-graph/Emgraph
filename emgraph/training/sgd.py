import math

import tensorflow as tf

from emgraph.training.utils import export_emgraph_optimizer
from emgraph.training._optimizer_constants import DEFAULT_LR, DEFAULT_DECAY_CYCLE, DEFAULT_DECAY_CYCLE_MULTIPLE, \
    DEFAULT_LR_DECAY_FACTOR, DEFAULT_END_LR, DEFAULT_SINE
from emgraph.training.optimizer import Optimizer


@export_emgraph_optimizer("sgd", ['lr', 'decay_cycle', 'end_lr', 'sine_decay', 'expand_factor', 'decay_lr_rate'])
class SGD(Optimizer):
    """
    SGD Optimizer.

    """

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

        super(SGD, self).__init__(optimizer_params, batches_count, verbose)

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

    def minimize(self, loss, var_list):
        """
        Create an optimizer to minimize the model loss.

        :param loss: Loss node for computing model loss.
        :type loss: tf.Tensor
        :param var_list: A list of variables to train
        :type var_list: list
        :return: Node that needs to be evaluated for minimizing the loss during training
        :rtype: tf.Operation
        """

        # create a placeholder for learning rate
        # self.lr_placeholder = tf.placeholder(tf.float32)
        self.lr_placeholder = 0

        # create the optimizer with the placeholder
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_placeholder)
        self.optimizer = tf.optimizers.SGD(learning_rate=self._optimizer_params['lr'],
                                           momentum=self._optimizer_params['momentum'])

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
        train = self.optimizer.minimize(loss, var_list)
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

        # Sinusoidal Decay
        if self.is_cosine_decay:
            # compute the cycle number
            current_cycle_num = \
                ((epoch_num - 1 - self.curr_start) * self.batches_count + (batch_num - 1)) / \
                (self.curr_cycle_length * self.batches_count)
            # compute a learning rate for the current batch/epoch
            self.current_lr = \
                self.end_lr + (self.start_lr - self.end_lr) * 0.5 * (1 + math.cos(math.pi * current_cycle_num))

            # Start the next cycle and Expand the cycle/Decay the learning rate
            if epoch_num % (self.next_cycle_epoch - 1) == 0 and batch_num == self.batches_count:
                self.curr_cycle_length = self.curr_cycle_length * self.decay_cycle_expand_factor
                self.next_cycle_epoch = self.next_cycle_epoch + self.curr_cycle_length
                self.curr_start = epoch_num
                self.start_lr = self.start_lr / self.decay_lr_rate

            if self.current_lr < self.end_lr:
                self.current_lr = self.end_lr

        # fixed rate decay
        elif self.decay_cycle_rate > 0:
            if epoch_num % (self.next_cycle_epoch) == 0 and batch_num == 1:
                if self.current_lr > self.end_lr:
                    self.next_cycle_epoch = self.decay_cycle_rate + \
                                            ((self.next_cycle_epoch - 1) * self.decay_cycle_expand_factor) + 1
                    self.current_lr = self.current_lr / self.decay_lr_rate

                    if self.current_lr < self.end_lr:
                        self.current_lr = self.end_lr

        # no change to the learning rate
        else:
            pass

        feed_dict.update({self.lr_placeholder: self.current_lr})