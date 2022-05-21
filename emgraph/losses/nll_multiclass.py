import tensorflow as tf

from emgraph.losses.loss import Loss
from emgraph.losses.utils import clip_before_exp, export_emgraph_loss


@export_emgraph_loss("multiclass_nll", [], {"require_same_size_pos_neg": False})
class NLLMulticlass(Loss):
    r"""
    Multiclass NLL loss introduced in :cite:`chen2015` where both the subject and objects are corrupted (to use it in
    this way pass corrupt_sides = ['s', 'o'] to embedding_model_params).

    This loss was re-engineered in :cite:`kadlecBK17` where only the object was corrupted to get improved
    performance (to use it in this way pass corrupt_sides = 'o' to embedding_model_params).

    .. math::

        \mathcal{L(X)} = -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_2|e_1,r_k)
         -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_1|r_k, e_2)

    Examples
    --------
    >>> from emgraph.models import TransE
    >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10,
    >>>                embedding_model_params={'corrupt_sides':['s', 'o']},
    >>>                loss='multiclass_nll', hyperparam_dict={})
    """

    def __init__(self, eta, hyperparam_dict=None, verbose=False):
        """
        Initialize the NLL Multiclass class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary
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
        :return: -
        :rtype: -
        """

        pass

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor, shape [n, 1]
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor, shape [n * negative_count, 1]
        :return: The loss value that is going to be minimized
        :rtype: float
        """

        # Fix for numerical instability of multiclass loss
        scores_pos = clip_before_exp(scores_pos)
        scores_neg = clip_before_exp(scores_neg)

        scores_neg_reshaped = tf.reshape(
            scores_neg, [self._loss_parameters["eta"], tf.shape(scores_pos)[0]]
        )
        neg_exp = tf.exp(scores_neg_reshaped)
        pos_exp = tf.exp(scores_pos)
        softmax_score = pos_exp / (tf.reduce_sum(neg_exp, axis=0) + pos_exp)

        loss = -tf.reduce_sum(tf.math.log(softmax_score))
        return loss
