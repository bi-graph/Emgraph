import tensorflow as tf

from emgraph.losses._loss_constants import DEFAULT_MARGIN
from emgraph.losses.loss import Loss
from emgraph.losses.utils import export_emgraph_loss


@export_emgraph_loss("pairwise", ["margin"])
class PairwiseLoss(Loss):
    r"""
    Pairwise, max-margin loss. :cite:`bordes2013translating`.
    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}}max(0, [\gamma + f_{model}(t^-;\Theta)
         - f_{model}(t^+;\Theta)])

    where :math:`\gamma` denotes the margin, :math:`\mathcal{G}` stands for the set of positives,
    :math:`\mathcal{C}` shows the set of corruptions and :math:`f_{model}(t;\Theta)` is the model-specific
    scoring function.
    """

    def __init__(self, eta, hyperparam_dict=None, verbose=False):
        """
        Initialize the Loss class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary

            - **'margin'**: (float). Margin to be used in pairwise loss computation (default: 1)

            Example: ``hyperparam_dict={'margin': 0}``
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        if hyperparam_dict is None:
            hyperparam_dict = {"margin": DEFAULT_MARGIN}
        super().__init__(eta, hyperparam_dict, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
            - **margin** - Margin to be used in pairwise loss computation(default:1)
        :type hyperparam_dict: dict
        """

        self._loss_parameters["margin"] = hyperparam_dict.get("margin", DEFAULT_MARGIN)

    # TODO: rename this to call or something else
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

        margin = tf.constant(
            self._loss_parameters["margin"], dtype=tf.float32, name="margin"
        )
        loss = tf.reduce_sum(tf.maximum(margin - scores_pos + scores_neg, 0))
        return loss
