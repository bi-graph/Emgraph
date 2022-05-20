import tensorflow as tf

from emgraph.losses._loss_constants import DEFAULT_ALPHA_ADVERSARIAL, DEFAULT_MARGIN_ADVERSARIAL
from emgraph.losses.loss import Loss
from emgraph.losses.utils import export_emgraph_loss


@export_emgraph_loss("self_adversarial", ['margin', 'alpha'], {'require_same_size_pos_neg': False})
class SelfAdversarialLoss(Loss):
    r"""
    Self adversarial sampling loss :cite:`sun2018rotate`.
    .. math::

        \mathcal{L} = -log\, \sigma(\gamma + f_{model} (\mathbf{s},\mathbf{o}))
        - \sum_{i=1}^{n} p(h_{i}^{'}, r, t_{i}^{'} ) \ log \
        \sigma(-f_{model}(\mathbf{s}_{i}^{'},\mathbf{o}_{i}^{'}) - \gamma)

    where :math:`\mathbf{s}, \mathbf{o} \in \mathcal{R}^k` are the embeddings of the subject
    and object of a triple :math:`t=(s,r,o)`, :math:`\gamma` shows the margin, :math:`\sigma` denotes the sigmoid function,
    and :math:`p(s_{i}^{'}, r, o_{i}^{'} )` is the negatives sampling distribution which is defined as:

    .. math::

        p(s'_j, r, o'_j | \{(s_i, r_i, o_i)\}) = \frac{\exp \alpha \, f_{model}(\mathbf{s'_j}, \mathbf{o'_j})}
        {\sum_i \exp \alpha \, f_{model}(\mathbf{s'_i}, \mathbf{o'_i})}

    where :math:`\alpha` is the temperature of sampling, :math:`f_{model}` is the scoring function of
    the desired embeddings model.
    """

    def __init__(self, eta, hyperparam_dict=None, verbose=False):
        """
        Initialize the Self adversarial sampling Loss class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary

            - **'margin'**: (float). Margin to be used in pairwise loss computation (default: 1)
            - **'alpha'** : (float). Temperature of sampling (default:0.5)
            Example: ``hyperparam_dict={'margin': 1, 'alpha': 0.5}``
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        if hyperparam_dict is None:
            hyperparam_dict = {'margin': DEFAULT_MARGIN_ADVERSARIAL, 'alpha': DEFAULT_ALPHA_ADVERSARIAL}
        super().__init__(eta, hyperparam_dict, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
            - **margin** - Margin to be used in pairwise loss computation(default:1)
            - **alpha** - Temperature of sampling (default:0.5)
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN_ADVERSARIAL)
        self._loss_parameters['alpha'] = hyperparam_dict.get('alpha', DEFAULT_ALPHA_ADVERSARIAL)

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

        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        alpha = tf.constant(self._loss_parameters['alpha'], dtype=tf.float32, name='alpha')

        # Compute p(neg_samples) based on eq 4
        scores_neg_reshaped = tf.reshape(scores_neg, [self._loss_parameters['eta'], tf.shape(scores_pos)[0]])
        p_neg = tf.nn.softmax(alpha * scores_neg_reshaped, axis=0)

        # Compute Loss based on eg 5
        loss = tf.reduce_sum(-tf.math.log_sigmoid(margin - tf.negative(scores_pos))) - tf.reduce_sum(
            tf.multiply(p_neg, tf.math.log_sigmoid(tf.negative(scores_neg_reshaped) - margin))
        )

        return loss
