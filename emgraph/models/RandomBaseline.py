import tensorflow as tf

from emgraph.utils import constants as constants
from .EmbeddingModel import EmbeddingModel, register_model

tf.device("/physical_device:GPU:0")  # todo: fix me


@register_model("RandomBaseline")
class RandomBaseline(EmbeddingModel):
    """Random baseline

    A dummy model that generates a pseudo-random score between 0 and 1 from a uniform distribution.

    This model is useful when comparing the performance of another model on a custom knowledge graph when no other
    baseline is available.

    .. note:: Although the model still requires the "fit()" function to be used, no real training will take place.

    Examples:

    >>> import numpy as np
    >>> from emgraph.models import RandomBaseline
    >>> model = RandomBaseline()
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> model.fit(X)
    >>> model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    [0.5488135039273248, 0.7151893663724195]
    """

    def __init__(self, seed=constants.DEFAULT_SEED, verbose=constants.DEFAULT_VERBOSE):
        """Initialize the RandomBaseline class

        :param seed: The seed used by the internal random numbers generator.
        :type seed: int
        :param verbose: Verbose mode.
        :type verbose: bool
        """

        super().__init__(
            k=1, eta=1, epochs=1, batches_count=1, seed=seed, verbose=verbose
        )
        self.all_params = {"seed": seed, "verbose": verbose}

    def _fn(self, e_s, e_p, e_o):
        """Random baseline scoring function: random number between 0 and 1.

        :param e_s: The embeddings of a list of subjects.
        :type e_s: tf.Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: tf.Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: tf.Tensor, shape [n]
        :return: Random number between 0 and 1.
        :rtype: tf.Op
        """

        # During training TensorFlow requires that gradients with respect to the trainable variables exist
        if self.train_dataset_handle is not None:
            # Sigmoid reaches 1 quite quickly, so the `useless` variable below is 0 for all practical purposes
            useless = (
                tf.sigmoid(tf.reduce_mean(tf.clip_by_value(e_s, 1e10, 1e11))) - 1.0
            )
            return tf.random.uniform((tf.size(e_s),), minval=0, maxval=1) + useless
        else:
            return tf.random.uniform((tf.size(e_s),), minval=0, maxval=1)

    def fit(
        self,
        X,
        early_stopping=False,
        early_stopping_params={},
        focusE_numeric_edge_values=None,
        tensorboard_logs_path=None,
    ):
        """
        Train an EmbeddingModel (with optional early stopping).

        In actuality, there is no genuine training, and the early stopping criteria have no effect.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor
        :param early_stopping: Flag to enable early stopping (default:False).

        If set to ``True``, the training loop adopts the following early stopping heuristic:

        - The model will be trained regardless of early stopping for ``burn_in`` epochs.
        - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

        If such metric decreases for ``stop_interval`` checks, we stop training early.

        Note the metric is computed on ``x_valid``. This is usually a validation set that you held out.

        Furthermore, because "criteria" is a ranking metric, it necessitates the generation of negatives. Entities
        that cause corruptions can be defined, as long as the side(s) of a triple to corrupt are supplied. By
        supplying an array of positives to "x filter," the technique offers filtered metrics. This will be used to
        filter the on-the-fly negatives (i.e. the corruptions).

        .. note::

            Remember that the early halting criterion may impose some overhead (caused by the metric computation).
            The objective is to find a fair balance between such overhead and preserving training epochs.

            A common approach is to use MRR unfiltered: ::

                early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}

            Note the size of validation set also contributes to such overhead.
            In most cases a smaller validation set would be enough.
        :type early_stopping: bool
        :param early_stopping_params: Dictionary of hyperparameters for the early stopping heuristics.

        The following string keys are supported:

            - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
            - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
            - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered'
              early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr').
              Note this will affect training time (no filter by default).
            - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
            - **check_interval'**: int : Early stopping interval after burn-in (default:10).
            - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
            - **'corruption_entities'**: List of entities to be used for corruptions.
              If 'all', it uses all entities (default: 'all')
            - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

            Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``
        :type early_stopping_params: dict
        :param focusE_numeric_edge_values: Numeric values associated with links. Semantically, the numeric value can
        signify importance, uncertainity, significance, confidence, etc. If the numeric value is unknown pass a NaN
        weight. The model will uniformly randomly assign a numeric value. One can also think about assigning numeric
        values by looking at the distribution of it per predicate.
        :type focusE_numeric_edge_values: nd array (n, 1)
        :param tensorboard_logs_path: Path to store tensorboard logs, e.g. average training loss tracking per epoch
        (default: ``None`` indicating no logs will be collected). When provided it will create a folder under provided
        path and save tensorboard files there. To then view the loss in the terminal run: ``tensorboard --logdir
        <tensorboard_logs_path>``.
        :type tensorboard_logs_path: str or None
        Path to store tensorboard logs, e.g. average training loss tracking per epoch (default: ``None`` indicating
        no logs will be collected). When provided it will create a folder under provided path and save tensorboard
        files there. To then view the loss in the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
        :return:
        :rtype:
        """

        super().fit(
            X,
            early_stopping,
            early_stopping_params,
            focusE_numeric_edge_values,
            tensorboard_logs_path=tensorboard_logs_path,
        )

    def predict(self, X, from_idx=False):
        """
        Predict the scores of triples using a trained embedding model.

        The function returns raw scores generated by the model.

        .. note::
            To obtain probability estimates, calibrate the model with :func:`~EmbeddingModel.calibrate`,
            then call :func:`~DistMult.predict_proba`.

        :param X: The triples to score.
        :type X: ndarray, shape [n, 3]
        :param from_idx: If True, will skip conversion to internal IDs. (default: False).
        :type from_idx: bool
        :return: The predicted scores for input triples X.
        :rtype: ndarray, shape [n]
        """

        __doc__ = super().predict.__doc__  # NOQA
        return super().predict(X, from_idx=from_idx)

    def _calibrate(
        self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50
    ):
        """
        Calibrate predictions

        The method implements the heuristics described in :cite:`calibration`, using Platt scaling
        :cite:`platt1999probabilistic`.

        After calibration, the calibrated predictions may be accessed with:meth:'predict proba'.

        Calibration should ideally be done on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. "X pos" and "X neg" offer positive and negative triples, respectively. Because the optimization is
        performed using a second-order approach (limited-memory BFGS), no hyperparameters are required.

        #. Only positive triples are offered, while negative triples are formed by corruptions, as in training or
        evaluation. Because the optimization is performed using a first-order approach (ADAM), "batches count" and
        "epochs" must be supplied.


        The base rate of positive triples has a large influence on calibration. As a result, for mode (2) of
        operation, the user must supply the "positive base rate" option. The relative sizes of the positive and
        negative sets can automatically deduce mode (1). However, the user can override this by specifying a value
        for "positive base rate".

        When calibrating without negatives, the most difficult problem is defining the positive base rate. That is
        dependent on the user's selection of which triples will be assessed during testing Take WN11 as an example:
        it has around 50% positive triples on both the validation and test sets. As a result, the positive base rate
        is 50%. However, if the user resamples it to have 75% positives, Its prior calibration will be damaged if it
        contains more than 25% negatives. The user must now adjust the model using a The base rate is 75% positive.
        As a result, this parameter is affected by how the user interacts with the dataset. It is not possible to
        ascertain this mechanically or a priori.

        .. Note ::
            Incompatible with large graph mode (i.e. if ``self.dealing_with_large_graphs=True``).


        :param X_pos: Numpy array of positive triples.
        :type X_pos: ndarray (shape [n, 3])
        :param X_neg: Numpy array of negative triples.

            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.
        :type X_neg: ndarray (shape [n, 3])
        :param positive_base_rate: Base rate of positive statements.

            For example, if we assume there is a fifty-fifty chance of any query to be true, the base rate would be 50%.

            If ``X_neg`` is provided and this is `None`, the relative sizes of ``X_pos`` and ``X_neg`` will be used to
            determine the base rate. For example, if we have 50 positive triples and 200 negative triples,
            the positive base rate will be assumed to be 50/(50+200) = 1/5 = 0.2.

            This must be a value between 0 and 1.
        :type positive_base_rate: float
        :param batches_count: Number of batches to complete one epoch of the Platt scaling training.
            Only applies when ``X_neg`` is  `None`.
        :type batches_count: int
        :param epochs: Number of epochs used to train the Platt scaling model.
            Only applies when ``X_neg`` is  `None`.
        :type epochs: int
        :return:
        :rtype:

        Examples
        -------

        >>> import numpy as np
        >>> from sklearn.metrics import brier_score_loss, log_loss
        >>> from scipy.special import expit
        >>>
        >>> from emgraph.datasets import BaseDataset, DatasetType
        >>> from emgraph.models import RandomBaseline
        >>>
        >>> X = BaseDataset.load_dataset(DatasetType.WN11)
        >>> X_valid_pos = X['valid'][X['valid_labels']]
        >>> X_valid_neg = X['valid'][~X['valid_labels']]
        >>>
        >>> model = RandomBaseline(batches_count=64, seed=0, epochs=500, k=100, eta=20,
        >>>                optimizer='adam', optimizer_params={'lr':0.0001},
        >>>                loss='pairwise', verbose=True)
        >>>
        >>> model.fit(X['train'])
        >>>
        >>> # Raw scores
        >>> scores = model.predict(X['test'])
        >>>
        >>> # Calibrate with positives and negatives
        >>> model.calibrate(X_valid_pos, X_valid_neg, positive_base_rate=None)
        >>> probas_pos_neg = model.predict_proba(X['test'])
        >>>
        >>> # Calibrate with just positives and base rate of 50%
        >>> model.calibrate(X_valid_pos, positive_base_rate=0.5)
        >>> probas_pos = model.predict_proba(X['test'])
        >>>
        >>> # Calibration evaluation with the Brier score loss (the smaller, the better)
        >>> print("Brier scores")
        >>> print("Raw scores:", brier_score_loss(X['test_labels'], expit(scores)))
        >>> print("Positive and negative calibration:", brier_score_loss(X['test_labels'], probas_pos_neg))
        >>> print("Positive only calibration:", brier_score_loss(X['test_labels'], probas_pos))
        Brier scores
        Raw scores: 0.4925058891371126
        Positive and negative calibration: 0.20434617882733366
        Positive only calibration: 0.22597599585144656
        """
        __doc__ = super()._calibrate.__doc__  # NOQA
        super()._calibrate(X_pos, X_neg, positive_base_rate, batches_count, epochs)

    def _predict_proba(self, X):
        """
        Predicts probabilities using the Platt scaling model (after calibration).

        Model must be calibrated beforehand with the ``calibrate`` method.

        :param X: Numpy array of triples to be evaluated.
        :type X: ndarray, shape [n, 3]
        :return: Probability of each triple to be true according to the Platt scaling calibration.
        :rtype: ndarray, shape [n, 3]
        """
        __doc__ = super()._predict_proba.__doc__  # NOQA
        return super()._predict_proba(X)
