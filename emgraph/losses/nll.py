import tensorflow as tf

from emgraph.losses.utils import export_emgraph_loss, clip_before_exp
from emgraph.losses.loss import Loss


@export_emgraph_loss("nll")
class NLLLoss(Loss):
    r"""
    Negative log-likelihood loss :cite:`trouillon2016complex`.
    .. math::

        \mathcal{L}(\Theta) = \sum_{t \in \mathcal{G} \cup \mathcal{C}}log(1 + exp(-y \, f_{model}(t;\Theta)))

    where :math:`y \in {-1, 1}` denotes the label of the statement, :math:`\mathcal{G}` stands for the set of positives,
    :math:`\mathcal{C}` shows the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.
    """

    def __init__(self, eta, hyperparam_dict=None, verbose=False):
        """
        Initialize NLL Loss class
        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary (non is required!)
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        if hyperparam_dict is None:
            hyperparam_dict = {}
        super().__init__(eta, hyperparam_dict, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        """
        pass

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor, shape [n, 1]
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor, shape [n, 1]
        :return: The loss value that is going to be minimized
        :rtype: float
        """

        scores_neg = clip_before_exp(scores_neg)
        scores_pos = clip_before_exp(scores_pos)
        scores = tf.concat([-scores_pos, scores_neg], 0)

        return tf.reduce_sum(tf.math.log(1 + tf.exp(scores)))