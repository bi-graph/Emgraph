import tensorflow as tf

from emgraph.losses._loss_constants import (
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LABEL_WEIGHTING,
    logger,
)
from emgraph.losses.loss import Loss
from emgraph.losses.utils import export_emgraph_loss


@export_emgraph_loss(
    "bce", ["label_smoothing", "label_weighting"], {"require_same_size_pos_neg": False}
)
class BCELoss(Loss):
    r"""
    Binary Cross Entropy Loss.

    .. math::

        \mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} y_i \cdot log(p(y_i)) + (1-y_i) \cdot log(1-p(y_i))

    Examples
    --------
    >>> from emgraph.models import ConvE
    >>> model = ConvE(batches_count=1, seed=555, epochs=20, k=10, loss='bce', hyperparam_dict={})
    """

    def __init__(self, eta, hyperparam_dict={}, verbose=False):
        """
        Initialize the BCE loss class.

        :param eta: Number of negatives
        :type eta: int
        :param hyperparam_dict: Hyperparameters dictionary
        :type hyperparam_dict: dict
        :param verbose: Set / unset verbose mode
        :type verbose: bool
        """

        super().__init__(eta, hyperparam_dict, verbose)

    def _inputs_check(self, y_true, y_pred):
        """
        Check and create dependencies needed by loss computations.

        :param y_true: A tensor of ground truth values.
        :type y_true: tf.Tensor
        :param y_pred: A tensor of predicted values.
        :type y_pred: tf.Tensor
        :return: -
        :rtype: -
        """

        logger.debug("Creating dependencies before loss computations.")

        self._dependencies = []
        logger.debug("Dependencies found: \n\tRequired same size y_true and y_pred. ")
        self._dependencies.append(
            tf.Assert(
                tf.equal(tf.shape(y_pred)[0], tf.shape(y_true)[0]),
                [tf.shape(y_pred)[0], tf.shape(y_true)[0]],
            )
        )

        if self._loss_parameters["label_smoothing"] is not None:
            if "num_entities" not in self._loss_parameters.keys():
                msg = (
                    "To apply label smooth-ing the number of entities must be known. "
                    "Set using '_set_hyperparams('num_entities', value)'."
                )
                logger.error(msg)
                raise Exception(msg)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize, Verify and Store the hyperparameters.

        :param hyperparam_dict: Key-value dictionary for hyperparameters.
            - **label_smoothing** (float): Apply label smoothing to vector of true labels. Can improve multi-class
            classification training by using soft targets that are a weighted average of hard targets and the
            uniform distribution over labels. Default: None
            - **label_weighting** (bool): Apply label weighting to vector of true labels. Gives lower weight to
            outputs with more positives in one-hot vector. Default: False
        :type hyperparam_dict: dict
        :return: -
        :rtype: -
        """

        self._loss_parameters["label_smoothing"] = hyperparam_dict.get(
            "label_smoothing", DEFAULT_LABEL_SMOOTHING
        )
        self._loss_parameters["label_weighting"] = hyperparam_dict.get(
            "label_weighting", DEFAULT_LABEL_WEIGHTING
        )

    def _set_hyperparams(self, key, value):
        """
        Set the hyperparameters for the loss function.

        :param key: Key for the hyperparameters dictionary
        :type key: str/int
        :param value: Value for the hyperparameters dictionary
        :type value: str/int
        :return: -
        :rtype: -
        """

        if key in self._loss_parameters.keys():
            msg = (
                "{} already exists in loss hyperparameters dict with value {} \n"
                "Overriding with value {}.".format(
                    key, self._loss_parameters[key], value
                )
            )
            logger.info(msg)

        self._loss_parameters[key] = value

    def apply(self, y_true, y_pred):
        """
        Interface of the Loss class. Check, preprocess the inputs and apply the loss function.

        :param y_true: Ground truth values tensor
        :type y_true: tf.Tensor
        :param y_pred: Predicted values tensor
        :type y_pred: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: tf.Tensor
        """

        self._inputs_check(y_true, y_pred)
        with tf.control_dependencies(self._dependencies):
            loss = self._apply(y_true, y_pred)
        return loss

    def _apply(self, y_true, y_pred):
        """
        Apply the loss function.

        :param y_true: Ground truth values tensor
        :type y_true: tf.Tensor
        :param y_pred: Predicted values tensor
        :type y_pred: tf.Tensor
        :return: The loss value that is going to be minimized
        :rtype: float
        """

        if self._loss_parameters["label_smoothing"] is not None:
            y_true = tf.add(
                (1 - self._loss_parameters["label_smoothing"]) * y_true,
                (self._loss_parameters["label_smoothing"])
                / self._loss_parameters["num_entities"],
            )

        if self._loss_parameters["label_weighting"]:

            eps = 1e-6
            wt = tf.reduce_mean(y_true)
            loss = -tf.reduce_sum(
                (1 - wt) * y_true * tf.math.log_sigmoid(y_pred)
                + wt * (1 - y_true) * tf.math.log(1 - tf.sigmoid(y_pred) + eps)
            )

        else:
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            )

        return loss
