
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
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
        :type hyperparam_dict: dict
        :return:
        :rtype:
        """

        msg = 'This function is a placeholder in an abstract class'
        logger.error(msg)
        NotImplementedError(msg)

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

    def _inputs_check(self, scores_pos, scores_neg):
        """
        Check and create dependencies needed by loss computations.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        """

        logger.debug('Creating dependencies before loss computations.')
        self._dependencies = []
        if LOSS_REGISTRY[self.name].class_params['require_same_size_pos_neg'] and self._loss_parameters['eta'] != 1:
            logger.debug('Dependencies found: \n\tRequired same size positive and negative. \n\tEta is not 1.')
            self._dependencies.append(tf.Assert(tf.equal(tf.shape(scores_pos)[0], tf.shape(scores_neg)[0]),
                                                [tf.shape(scores_pos)[0], tf.shape(scores_neg)[0]]))

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss-function. All child classes must override this method.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        msg = 'This function is a placeholder in an abstract class.'
        logger.error(msg)
        NotImplementedError(msg)

    def apply(self, scores_pos, scores_neg):
        """
        Interface of the Loss class. Check, preprocess the inputs and apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        self._inputs_check(scores_pos, scores_neg)
        with tf.control_dependencies(self._dependencies):
            loss = self._apply(scores_pos, scores_neg)
        return loss


@register_loss("pairwise", ['margin'])
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
            hyperparam_dict = {'margin': DEFAULT_MARGIN}
        super().__init__(eta, hyperparam_dict, verbose)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
            - **margin** - Margin to be used in pairwise loss computation(default:1)
        :type hyperparam_dict: dict
        """

        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN)

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor, shape [n, 1]
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor, shape [n, 1]
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        loss = tf.reduce_sum(tf.maximum(margin - scores_pos + scores_neg, 0))
        return loss


@register_loss("nll")
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
        :rtype: tf.Tensor
        """

        scores_neg = clip_before_exp(scores_neg)
        scores_pos = clip_before_exp(scores_pos)
        scores = tf.concat([-scores_pos, scores_neg], 0)
        return tf.reduce_sum(tf.log(1 + tf.exp(scores)))

@register_loss("absolute_margin", ['margin'])
class AbsoluteMarginLoss(Loss):
    r"""
    Absolute margin, max-margin loss :cite:`Hamaguchi2017`.
    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}} f_{model}(t^-;\Theta)
        - max(0, [\gamma - f_{model}(t^+;\Theta)])

    where :math:`\gamma` denotes the margin, :math:`\mathcal{G}` stands for the set of positives, :math:`\mathcal{C}`
    shows the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.
    """

    def __init__(self, eta, hyperparam_dict=None, verbose=False):
        """
        Initialize the Absolute Margin Loss class.

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
            hyperparam_dict = {'margin': DEFAULT_MARGIN}
        super().__init__(eta, hyperparam_dict, verbose)


    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
            - **margin** - Margin to be used in pairwise loss computation(default:1)
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN)

    def _apply(self, scores_pos, scores_neg):
        """
        Apply the loss function.

        :param scores_pos: A tensor of scores assigned to the positive statements
        :type scores_pos: tf.Tensor, shape [n, 1]
        :param scores_neg: A tensor of scores assigned to the negative statements
        :type scores_neg: tf.Tensor, shape [n, 1]
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        loss = tf.reduce_sum(tf.maximum(margin + scores_neg, 0) - scores_pos)
        return loss


@register_loss("self_adversarial", ['margin', 'alpha'], {'require_same_size_pos_neg': False})
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
        :rtype: tf.Tensor
        """

        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        alpha = tf.constant(self._loss_parameters['alpha'], dtype=tf.float32, name='alpha')

        # Compute p(neg_samples) based on eq 4
        scores_neg_reshaped = tf.reshape(scores_neg, [self._loss_parameters['eta'], tf.shape(scores_pos)[0]])
        p_neg = tf.nn.softmax(alpha * scores_neg_reshaped, axis=0)

        # Compute Loss based on eg 5
        loss = tf.reduce_sum(-tf.log_sigmoid(margin - tf.negative(scores_pos))) - tf.reduce_sum(
            tf.multiply(p_neg, tf.log_sigmoid(tf.negative(scores_neg_reshaped) - margin)))

        return loss

@register_loss("multiclass_nll", [], {'require_same_size_pos_neg': False})
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
    >>> from bigraph.latent_features import TransE
    >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10,
    >>>                embedding_model_params={'corrupt_sides':['s', 'o']},
    >>>                loss='multiclass_nll', loss_params={})
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
        :rtype: tf.Tensor
        """

        # Fix for numerical instability of multiclass loss
        scores_pos = clip_before_exp(scores_pos)
        scores_neg = clip_before_exp(scores_neg)

        scores_neg_reshaped = tf.reshape(scores_neg, [self._loss_parameters['eta'], tf.shape(scores_pos)[0]])
        neg_exp = tf.exp(scores_neg_reshaped)
        pos_exp = tf.exp(scores_pos)
        softmax_score = pos_exp / (tf.reduce_sum(neg_exp, axis=0) + pos_exp)

        loss = -tf.reduce_sum(tf.log(softmax_score))
        return loss
