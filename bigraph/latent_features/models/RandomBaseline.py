
import tensorflow as tf

from .EmbeddingModel import EmbeddingModel, register_model
from bigraph.latent_features import constants as constants

@register_model("RandomBaseline")
class RandomBaseline(EmbeddingModel):
    """Random baseline

    A dummy model that assigns a pseudo-random score included between 0 and 1,
    drawn from a uniform distribution.

    The model is useful whenever you need to compare the performance of
    another model on a custom knowledge graph, and no other baseline is available.

    .. note:: Although the model still requires invoking the ``fit()`` method,
        no actual training will be carried out.

    Examples:

    >>> import numpy as np
    >>> from bigraph.latent_features import RandomBaseline
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

        super().__init__(k=1, eta=1, epochs=1, batches_count=1, seed=seed, verbose=verbose)
        self.all_params = \
            {
                'seed': seed,
                'verbose': verbose
            }

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
            useless = tf.sigmoid(tf.reduce_mean(tf.clip_by_value(e_s, 1e10, 1e11))) - 1.0
            return tf.random_uniform((tf.size(e_s),), minval=0, maxval=1) + useless
        else:
            return tf.random_uniform((tf.size(e_s),), minval=0, maxval=1)

    def fit(self, X, early_stopping=False, early_stopping_params={}, focusE_numeric_edge_values=None,
            tensorboard_logs_path=None):
        """Train an EmbeddingModel (with optional early stopping).
        There is no actual training involved in practice and the early stopping parameters won't have any effect.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of BigraphDatasetAdapter
        :param early_stopping: Flag to enable early stopping (default:False).

        If set to ``True``, the training loop adopts the following early stopping heuristic:

        - The model will be trained regardless of early stopping for ``burn_in`` epochs.
        - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

        If such metric decreases for ``stop_interval`` checks, we stop training early.

        Note the metric is computed on ``x_valid``. This is usually a validation set that you held out.

        Also, because ``criteria`` is a ranking metric, it requires generating negatives.
        Entities used to generate corruptions can be specified, as long as the side(s) of a triple to corrupt.
        The method supports filtered metrics, by passing an array of positives to ``x_filter``. This will be used to
        filter the negatives generated on the fly (i.e. the corruptions).

        .. note::

            Keep in mind the early stopping criteria may introduce a certain overhead
            (caused by the metric computation).
            The goal is to strike a good trade-off between such overhead and saving training epochs.

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

        super().fit(X, early_stopping, early_stopping_params, focusE_numeric_edge_values,
                    tensorboard_logs_path=tensorboard_logs_path)

    def predict(self, X, from_idx=False):
        """Predict the scores of triples using a trained embedding model.
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