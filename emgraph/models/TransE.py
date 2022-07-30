import tensorflow as tf

from emgraph.initializers._initializer_constants import DEFAULT_GLOROT_IS_UNIFORM
from emgraph.utils import constants as constants
from .EmbeddingModel import EmbeddingModel, register_model

tf.device("/physical_device:GPU:0")  # todo: fix me


@register_model("TransE", ["norm", "normalize_ent_emb", "negative_corruption_entities"])
class TransE(EmbeddingModel):
    r"""
    Translating Embeddings (TransE)

    The model as described in :cite:`bordes2013translating`.

    The scoring function of TransE computes a similarity between the embedding of the subject
    :math:`\mathbf{e}_{sub}` translated by the embedding of the predicate :math:`\mathbf{e}_{pred}`
    and the embedding of the object :math:`\mathbf{e}_{obj}`,
    using the :math:`L_1` or :math:`L_2` norm :math:`||\cdot||`:

    .. math::

        f_{TransE}=-||\mathbf{e}_{sub} + \mathbf{e}_{pred} - \mathbf{e}_{obj}||_n


    Such scoring function is then used on positive and negative triples :math:`t^+, t^-` in the loss function.

    :param k: Embedding space dimensionality
    :type k: int
    :param eta: The number of negatives that must be generated at runtime during training for each positive.
    :type eta: int
    :param epochs: The iterations of the training loop.
    :type epochs: int
    :param batches_count: The number of batches in which the training set must be split during the training loop.
    :type batches_count: int
    :param seed:The seed used by the internal random numbers generator.
    :type seed: int
    :param embedding_model_params: TransE-specific hyperparams, passed to the model as a dictionary.

        Supported keys:

        - **'normalize_ent_emb'** (bool): flag to indicate whether to normalize entity embeddings
          after each batch update (default: False).
        - **'negative_corruption_entities'** - Entities to be used for generation of corruptions while training.
          It can take the following values :
          ``all`` (default: all entities),
          ``batch`` (entities present in each batch),
          list of entities
          or an int (which indicates how many entities that should be used for corruption generation).
        - **corrupt_sides** : Specifies how to generate corruptions for training.
          Takes values `s`, `o`, `s+o` or any combination passed as a list
        - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
        - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value
        until it reachs original value.
        - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph
        topology.
        - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]

        The last 4 parameters are related to FocusE layers.

        Example: ``embedding_model_params={'normalize_ent_emb': False}``
    :type embedding_model_params: dict
    :param optimizer: The optimizer used to minimize the loss function. Choose between 'sgd',
        'adagrad', 'adam', 'momentum'.
    :type optimizer: str
    :param optimizer_params: Arguments specific to the optimizer, passed as a dictionary.

        Supported keys:

        - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
        - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

        Example: ``optimizer_params={'lr': 0.01}``
    :type optimizer_params: dict
    :param loss: The type of loss function to use during training.

        - ``pairwise``  the model will use pairwise margin-based loss function.
        - ``nll`` the model will use negative loss likelihood.
        - ``absolute_margin`` the model will use absolute margin likelihood.
        - ``self_adversarial`` the model will use adversarial sampling loss function.
        - ``multiclass_nll`` the model will use multiclass nll loss.
          Switch to multiclass loss defined in :cite:`chen2015` by passing 'corrupt_sides'
          as ['s','o'] to embedding_model_params.
          To use loss defined in :cite:`kadlecBK17` pass 'corrupt_sides' as 'o' to embedding_model_params.
    :type loss: str
    :param loss_params: Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>`
        documentation for additional details.

        Example: ``optimizer_params={'lr': 0.01}`` if ``loss='pairwise'``.
    :type loss_params: dict
    :param regularizer: The regularization strategy to use with the loss function.

        - ``None``: the model will not use any regularizer (default)
        - 'LP': the model will use L1, L2 or L3 based on the value of ``regularizer_params['p']`` (see below).
    :type regularizer: str
    :param regularizer_params: Dictionary of regularizer-specific hyperparameters. See the :ref:`regularizers <ref-reg>`
        documentation for additional details.

        Example: ``regularizer_params={'lambda': 1e-5, 'p': 2}`` if ``regularizer='LP'``.
    :type regularizer_params: dict
    :param initializer: The type of initializer to use.

        - ``normal``: The embeddings will be initialized from a normal distribution
        - ``uniform``: The embeddings will be initialized from a uniform distribution
        - ``xavier``: The embeddings will be initialized using xavier strategy (default)
    :type initializer: str
    :param initializer_params: Dictionary of initializer-specific hyperparameters. See the
        :ref:`initializer <ref-init>`
        documentation for additional details.

        Example: ``initializer_params={'mean': 0, 'std': 0.001}`` if ``initializer='normal'``.
    :type initializer_params: dict
    :param verbose: Verbose mode
    :type verbose: bool

    Examples:

    >>> import numpy as np
    >>> from emgraph.models import TransE
    >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin':5})
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
    [-4.6903257, -3.9047198]
    >>> model.get_embeddings(['f','e'], embedding_type='entity')
    array([[ 0.10673896, -0.28916815,  0.6278883 , -0.1194713 , -0.10372276,
    -0.37258488,  0.06460134, -0.27879423,  0.25456288,  0.18665907],
    [-0.64494324, -0.12939683,  0.3181001 ,  0.16745451, -0.03766293,
     0.24314676, -0.23038973, -0.658638  ,  0.5680542 , -0.05401703]],
    dtype=float32)

    """

    def __init__(
        self,
        k=constants.DEFAULT_EMBEDDING_SIZE,
        eta=constants.DEFAULT_ETA,
        epochs=constants.DEFAULT_EPOCH,
        batches_count=constants.DEFAULT_BATCH_COUNT,
        seed=constants.DEFAULT_SEED,
        embedding_model_params={
            "norm": constants.DEFAULT_NORM_TRANSE,
            "normalize_ent_emb": constants.DEFAULT_NORMALIZE_EMBEDDINGS,
            "negative_corruption_entities": constants.DEFAULT_CORRUPTION_ENTITIES,
            "corrupt_sides": constants.DEFAULT_CORRUPT_SIDE_TRAIN,
        },
        optimizer=constants.DEFAULT_OPTIM,
        optimizer_params={"lr": constants.DEFAULT_LR},
        loss=constants.DEFAULT_LOSS,
        loss_params={},
        regularizer=constants.DEFAULT_REGULARIZER,
        regularizer_params={},
        initializer=constants.DEFAULT_INITIALIZER,
        initializer_params={"uniform": DEFAULT_GLOROT_IS_UNIFORM},
        verbose=constants.DEFAULT_VERBOSE,
        large_graphs=False,
    ):
        """
        Initialize an EmbeddingModel.

        Also creates a new Tensorflow session for training.
        """
        super().__init__(
            k=k,
            eta=eta,
            epochs=epochs,
            batches_count=batches_count,
            seed=seed,
            embedding_model_params=embedding_model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss,
            loss_params=loss_params,
            regularizer=regularizer,
            regularizer_params=regularizer_params,
            initializer=initializer,
            initializer_params=initializer_params,
            verbose=verbose,
            large_graphs=large_graphs,
        )

    def _fn(self, e_s, e_p, e_o):
        r"""
        The TransE scoring function.

        .. math::

            f_{TransE}=-||(\mathbf{e}_s + \mathbf{r}_p) - \mathbf{e}_o||_n

        :param e_s: The embeddings of a list of subjects.
        :type e_s: tf.Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: tf.Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: tf.Tensor, shape [n]
        :return: TransE scoring function operation
        :rtype: tf.Op
        """

        return tf.negative(
            tf.norm(
                e_s + e_p - e_o,
                ord=self.embedding_model_params.get(
                    "norm", constants.DEFAULT_NORM_TRANSE
                ),
                axis=1,
            )
        )

    def fit(
        self,
        X,
        early_stopping=False,
        early_stopping_params={},
        focusE_numeric_edge_values=None,
        tensorboard_logs_path=None,
    ):
        """
        Train a Translating Embeddings model.

        The model is trained on a training set X using the training protocol described in :cite:`trouillon2016complex`.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor
        :param early_stopping: Flag to enable early stopping (default:``False``)
            If set to ``True``, the training loop adopts the following early stopping heuristic:

                - The model will be trained regardless of early stopping for ``burn_in`` epochs.
                - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

            If such metric decreases for ``stop_interval`` checks, we stop training early.

            Note the metric is computed on ``x_valid``. This is usually a validation set that you held out.

            Furthermore, because "criteria" is a ranking metric, it necessitates the generation of negatives.
            Entities that cause corruptions can be defined, as long as the side(s) of a triple to corrupt are
            supplied. By supplying an array of positives to "x filter," the technique offers filtered metrics. This
            will be put to use to filter the on-the-fly created negatives (i.e. the corruptions).

            .. note::

                Remember that the early halting criterion may impose some overhead (caused by the metric
                computation). The objective is to find a fair balance between such overhead and preserving training
                epochs.

                A common approach is to use MRR unfiltered: ::

                    early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}

                It should be noted that the size of the validation set also adds to such cost. A smaller validation
                set would enough in most circumstances.

        :type early_stopping: bool
        :param early_stopping_params: Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor. Numpy array of validation triples OR handle of Dataset adapter which would help retrieve data.
                - **'criteria'**: str : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default). If the filter has already been set in the adapter, pass True
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **'check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o', 's,o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``
        :type early_stopping_params: dict
        :param focusE_numeric_edge_values: Numeric values associated with links.

            Semantically, a numerical number might represent relevance, uncertainty, significance, confidence,
            and so forth. If the numeric value is unknown, a NaN weight is applied. The model will assign a number
            value at random. One may also consider assigning numerical values by examining the distribution of them
            per condition.

            .. _focuse_transe:

            If processing a knowledge graph with numeric values associated with links, this is the vector of such
            numbers. Passing this argument will activate the :ref:`FocusE layer <edge-literals>` :cite:`pai2021learning`.

            Numeric values can represent importance, uncertainty, significance, confidence, and so forth. Values can
            be any number and will be automatically normalised to the [0, 1] range based on the criteria. If the
            numeric value is uncertain, use "np.NaN". The model will assign a number value at random.

            .. note::

                The following toy example shows how to enable the FocusE layer to process edges with numeric literals:
                ::

                    import numpy as np
                    from emgraph.models import TransE
                    model = TransE(batches_count=1, seed=555, epochs=20,
                                     k=10, loss='pairwise',
                                     loss_params={'margin':5})
                    X = np.array([['a', 'y', 'b'],
                                  ['b', 'y', 'a'],
                                  ['a', 'y', 'c'],
                                  ['c', 'y', 'a'],
                                  ['a', 'y', 'd'],
                                  ['c', 'y', 'd'],
                                  ['b', 'y', 'c'],
                                  ['f', 'y', 'e']])

                    # Numeric values below are associate to each triple in X.
                    # They can be any number and will be automatically
                    # normalised to the [0, 1] range, on a
                    # predicate-specific basis.
                    X_edge_values = np.array([5.34, -1.75, 0.33, 5.12,
                                              np.nan, 3.17, 2.76, 0.41])

                    model.fit(X, focusE_numeric_edge_values=X_edge_values)
        :type focusE_numeric_edge_values: nd array (n, 1)
        :param tensorboard_logs_path: Path to store tensorboard logs, e.g. average training loss tracking per epoch (default: ``None`` indicating
            no logs will be collected). When provided it will create a folder under provided path and save tensorboard
            files there. To then view the loss in the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
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

        .. note:: To obtain probability estimates, calibrate the model with :func:`~EmbeddingModel.calibrate`,
        then call :func:`~TrnasE.predict_proba`.

        :param X: The triples to score.
        :type X: ndarray, shape [n, 3]
        :param from_idx: If True, will skip conversion to internal IDs. (default: False).
        :type from_idx: bool
        :return: The predicted scores for input triples X.
        :rtype: ndarray, shape [n]
        """
        # __doc__ = super().predict.__doc__  # NOQA
        return super().predict(X, from_idx=from_idx)

    def _calibrate(
        self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50
    ):
        __doc__ = super()._calibrate.__doc__  # NOQA
        super()._calibrate(X_pos, X_neg, positive_base_rate, batches_count, epochs)

    def _predict_proba(self, X):
        __doc__ = super()._predict_proba.__doc__  # NOQA
        return super()._predict_proba(X)
