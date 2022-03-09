
from .EmbeddingModel import EmbeddingModel, register_model
from bigraph.latent_features import constants as constants
from bigraph.latent_features.initializers import DEFAULT_XAVIER_IS_UNIFORM
import tensorflow as tf


@register_model("DistMult",
                ["normalize_ent_emb", "negative_corruption_entities"])
class DistMult(EmbeddingModel):
    r"""The DistMult model :cite:`yang2014embedding`.


    The bilinear diagonal DistMult model uses the trilinear dot product as scoring function:

    .. math::

        f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \rangle

    where :math:`\mathbf{e}_{s}` is the embedding of the subject, :math:`\mathbf{r}_{p}` the embedding
    of the predicate and :math:`\mathbf{e}_{o}` the embedding of the object.

    Examples:

    >>> import numpy as np
    >>> from bigraph.latent_features import DistMult
    >>> model = DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
    >>>         loss_params={'margin':5})
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
    [-0.13863425, -0.09917116]
    >>> model.get_embeddings(['f','e'], embedding_type='entity')
    array([[ 0.10137264, -0.28248304,  0.6153027 , -0.13133956, -0.11675504,
    -0.37876177,  0.06027773, -0.26390398,  0.254603  ,  0.1888549 ],
    [-0.6467299 , -0.13729756,  0.3074872 ,  0.16966867, -0.04098966,
     0.25289047, -0.2212451 , -0.6527815 ,  0.5657673 , -0.03876532]],
    dtype=float32)

    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'normalize_ent_emb': constants.DEFAULT_NORMALIZE_EMBEDDINGS,
                                         'negative_corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
                                         'corrupt_sides': constants.DEFAULT_CORRUPT_SIDE_TRAIN},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize an DistMult

        Also creates a new Tensorflow session for training.

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
        :param embedding_model_params: DistMult-specific hyperparams, passed to the model as a dictionary.

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
        """

        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        r"""The scoring function of the DistMult.

            Assigns a score to a list of triples, with a model-specific strategy.
            Triples are passed as lists of subject, predicate, object embeddings.

                .. math::

                f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \rangle

            :param e_s: The embeddings of a list of subjects.
            :type e_s: tf.Tensor, shape [n]
            :param e_p: The embeddings of a list of predicates.
            :type e_p: tf.Tensor, shape [n]
            :param e_o: The embeddings of a list of objects.
            :type e_o: tf.Tensor, shape [n]
            :return: DistMult scoring function operation
            :rtype: tf.Op
            """


        return tf.reduce_sum(e_s * e_p * e_o, axis=1)

    def fit(self, X, early_stopping=False, early_stopping_params={},
            focusE_numeric_edge_values=None, tensorboard_logs_path=None):
        """Train an EmbeddingModel (with optional early stopping).

        The model is trained on a training set X using the training protocol
        described in :cite:`trouillon2016complex`.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of BigraphDatasetAdapter
        :param early_stopping: Flag to enable early stopping (default:``False``)
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

            - **'x_valid'**: ndarray (shape [n, 3]) or object of AmpligraphDatasetAdapter :
                             Numpy array of validation triples OR handle of Dataset adapter which
                             would help retrieve data.
            - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
            - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early
                              stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr').
                              Note this will affect training time (no filter by default).
                              If the filter has already been set in the adapter, pass True
            - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
            - **check_interval'**: int : Early stopping interval after burn-in (default:10).
            - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
            - **'corruption_entities'**: List of entities to be used for corruptions. If 'all',
              it uses all entities (default: 'all')
            - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o', 's,o' (default)

            Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``
        :type early_stopping_params: dict
        :param focusE_numeric_edge_values: Numeric values associated with links.
            Semantically, the numeric value can signify importance, uncertainity, significance, confidence, etc.
            If the numeric value is unknown pass a NaN weight. The model will uniformly randomly assign a numeric value.
            One can also think about assigning numeric values by looking at the distribution of it per predicate.

            .. _focuse_distmult:

        If processing a knowledge graph with numeric values associated with links, this is the vector of such
        numbers. Passing this argument will activate the :ref:`FocusE layer <edge-literals>`
        :cite:`pai2021learning`.
        Semantically, numeric values can signify importance, uncertainity, significance, confidence, etc.
        Values can be any number, and will be automatically normalised to the [0, 1] range, on a
        predicate-specific basis.
        If the numeric value is unknown pass a ``np.NaN`` value.
        The model will uniformly randomly assign a numeric value.

        .. note::

            The following toy example shows how to enable the FocusE layer
            to process edges with numeric literals: ::

                import numpy as np
                from ampligraph.latent_features import DistMult
                model = DistMult(batches_count=1, seed=555, epochs=20,
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

        super().fit(X, early_stopping, early_stopping_params, focusE_numeric_edge_values,
                    tensorboard_logs_path=tensorboard_logs_path)
