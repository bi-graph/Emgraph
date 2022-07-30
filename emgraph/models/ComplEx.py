import time

import tensorflow as tf

from emgraph.initializers._initializer_constants import DEFAULT_GLOROT_IS_UNIFORM
from emgraph.utils import constants as constants
from .EmbeddingModel import EmbeddingModel, register_model

tf.device("/physical_device:GPU:0")  # todo: fix me


@register_model("ComplEx", ["negative_corruption_entities"])
class ComplEx(EmbeddingModel):
    r"""Complex embeddings (ComplEx)

    The ComplEx model`[trouillon2016complex]` is an extension of the :class:`emgraph.models.ComplEx` bilinear
    diagonal model.

    ComplEx scoring function is based on the trilinear Hermitian dot product in :math:`\mathcal{C}`:

    .. math::

        f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

    ComplEx can be improved if used alongside the nuclear 3-norm (the **ComplEx-N3** model :cite:`lacroix2018canonical`),
    which can be easily added to the loss function via the ``regularizer`` hyperparameter with ``p=3`` and
    a chosen regularisation weight (represented by ``lambda``), as shown in the example below.

    See also :class:`emgraph.regularizers.LPRegularizer`.

    .. note::

        Since ComplEx embeddings belong to :math:`\mathcal{C}`, this model uses twice as many parameters as
        :class:`emgraph.models.ComplEx`.

    .. [trouillon2016complex] trouillon2016complex

    .. code:: python

        import numpy as np
        from emgraph.models import ComplEx
        model = ComplEx(
            batches_count=2,
            seed=555,
            epochs=100,
            k=20,
            eta=5,
            loss='pairwise',
            loss_params={'margin':1},
            regularizer='LP',
            regularizer_params={'p': 2, 'lambda':0.1})

    .. Examples::

    >>> import numpy as np
    >>> from emgraph.models import ComplEx
    >>>
    >>> model = ComplEx(batches_count=2, seed=555, epochs=100, k=20, eta=5,
    >>>             loss='pairwise', loss_params={'margin':1},
    >>>             regularizer='LP', regularizer_params={'p': 2, 'lambda':0.1})
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
    [[0.019520484], [-0.14998421]]
    >>> model.get_embeddings(['f','e'], embedding_type='entity')
    array([[-0.33021057,  0.26524785,  0.0446662 , -0.07932718, -0.15453218,
        -0.22342539, -0.03382565,  0.17444217,  0.03009969, -0.33569157,
         0.3200497 ,  0.03803705,  0.05536304, -0.00929996,  0.24446663,
         0.34408194,  0.16192885, -0.15033236, -0.19703785, -0.00783876,
         0.1495124 , -0.3578853 , -0.04975723, -0.03930473,  0.1663541 ,
        -0.24731971, -0.141296  ,  0.03150219,  0.15328223, -0.18549544,
        -0.39240393, -0.10824018,  0.03394471, -0.11075485,  0.1367736 ,
         0.10059565, -0.32808647, -0.00472086,  0.14231135, -0.13876757],
       [-0.09483694,  0.3531292 ,  0.04992269, -0.07774793,  0.1635035 ,
         0.30610007,  0.3666711 , -0.13785957, -0.3143734 , -0.36909637,
        -0.13792469, -0.07069954, -0.0368113 , -0.16743314,  0.4090072 ,
        -0.03407392,  0.3113114 , -0.08418448,  0.21435146,  0.12006859,
         0.08447982, -0.02025972,  0.38752195,  0.11451488, -0.0258422 ,
        -0.10990044, -0.22661531, -0.00478273, -0.0238297 , -0.14207476,
         0.11064807,  0.20135397,  0.22501846, -0.1731076 , -0.2770435 ,
         0.30784574, -0.15043163, -0.11599299,  0.05718031, -0.1300622 ]],
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
    ):
        """Initialize an EmbeddingModel

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
        :param embedding_model_params: ComplEx-specific hyperparams, passed to the model as a dictionary.

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
        )

        self.internal_k = self.k * 2

    def _initialize_parameters(self):
        """Initialize the complex embeddings.

        In this method, entity and relation embeddings (with size k) is created and initialized. If the graph is large,
        then only the required entity embeddings (max:batch_size*2) and all relation embeddings is loaded.
        """
        timestamp = int(time.time() * 1e6)
        if not self.dealing_with_large_graphs:
            self.ent_emb = self.make_variable(
                "ent_emb_{}".format(timestamp),
                shape=[len(self.ent_to_idx), self.internal_k],
                initializer=self.initializer.get_entity_initializer(
                    len(self.ent_to_idx), self.internal_k
                ),
                dtype=tf.float32,
            )
            self.rel_emb = self.make_variable(
                "rel_emb_{}".format(timestamp),
                shape=[len(self.rel_to_idx), self.internal_k],
                initializer=self.initializer.get_relation_initializer(
                    len(self.rel_to_idx), self.internal_k
                ),
                dtype=tf.float32,
            )
        else:
            # initialize entity embeddings to zero (these are reinitialized every batch by batch embeddings)
            self.ent_emb = self.make_variable(
                "ent_emb_{}".format(timestamp),
                shape=[self.batch_size * 2, self.internal_k],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32,
            )
            self.rel_emb = self.make_variable(
                "rel_emb_{}".format(timestamp),
                shape=[len(self.rel_to_idx), self.internal_k],
                initializer=self.initializer.get_relation_initializer(
                    len(self.rel_to_idx), self.internal_k
                ),
                dtype=tf.float32,
            )

    def _fn(self, e_s, e_p, e_o):
        r"""ComplEx scoring function.

        .. math::

            f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

        Additional details available in :cite:`trouillon2016complex` (Equation 9).

        :param e_s: The embeddings of a list of subjects.
        :type e_s: tf.Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: tf.Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: tf.Tensor, shape [n]
        :return: ComplEx scoring function operation
        :rtype: tf.Op
        """

        # Assume each embedding is made of an img and real component.
        # (These components are actually real numbers, see [trouillon2016complex].
        e_s_real, e_s_img = tf.split(e_s, 2, axis=1)
        e_p_real, e_p_img = tf.split(e_p, 2, axis=1)
        e_o_real, e_o_img = tf.split(e_o, 2, axis=1)

        # See Eq. 9 [trouillon2016complex):
        return (
            tf.reduce_sum(e_p_real * e_s_real * e_o_real, axis=1)
            + tf.reduce_sum(e_p_real * e_s_img * e_o_img, axis=1)
            + tf.reduce_sum(e_p_img * e_s_real * e_o_img, axis=1)
            - tf.reduce_sum(e_p_img * e_s_img * e_o_real, axis=1)
        )

    def fit(
        self,
        X,
        early_stopping=False,
        early_stopping_params={},
        focusE_numeric_edge_values=None,
        tensorboard_logs_path=None,
    ):
        """Train a ComplEx model.

        The model is trained on a training set X using the training protocol described in :cite:`trouillon2016complex`.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor
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

            Keep in mind the early stopping criteria may introduce a certain overhead (caused by the metric computation).
            The goal is to strike a good trade-off between such overhead and saving training epochs.

            A common approach is to use MRR unfiltered: ::

                early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}

            Notably, this overhead is also influenced by the size of the validation set. A smaller validation set
            would typically be sufficient.

        :type early_stopping: bool
        :param early_stopping_params: Dictionary of hyperparameters for the early stopping heuristics.

        The following string keys are supported:

            - **'x_valid'**: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor :
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
            Semantically, the numerical number might represent significance, confidence, relevance, and other concepts.
            Pass a NaN weight if the numerical amount is unknowable. The model will assign a numerical value uniformly at random.
            Assigning numerical values can also be considered by considering how they are distributed among predicates.

            .. _focuse_complex:

        If processing a knowledge graph with numeric values associated with links, this is the vector of such
        numbers. Passing this argument will activate the :ref:`FocusE layer <edge-literals>` :cite:`pai2021learning`.
        Semantically, the numerical number might represent significance, confidence, relevance, and other concepts.
        Values can be any number, and will be automatically normalised to the [0, 1] range, on a predicate-specific basis.
        The model will evenly and randomly assign a numeric value if the numeric value is unknown and is passed a value of "np.NaN".

        .. note::

            The following toy example shows how to enable the FocusE layer
            to process edges with numeric literals: ::

                import numpy as np
                from emgraph.models import ComplEx
                model = ComplEx(batches_count=1, seed=555, epochs=20,
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
        :param tensorboard_logs_path: Path to store tensorboard logs, e.g. average training loss tracking per epoch
        (default: ``None`` indicating no logs will be collected).
        When provided it will create a folder under provided path and save tensorboard files there.
        To then view the loss in the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
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

    def _calibrate(
        self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50
    ):
        """Calibrate predictions

        The method implements the heuristics described in :cite:`calibration`, using Platt scaling :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba` after calibration is done.

        Calibration should ideally be done on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. Both positive and negative triples are provided via ``X_pos`` and ``X_neg`` respectively. The optimization
        is done using a second-order method (limited-memory BFGS), therefore no hyperparameter needs to be specified.

        #. Only positive triples are provided, and the negative triples are generated by corruptions just like it is
        done in training or evaluation. The optimization is done using a first-order method (ADAM), therefore
        ``batches_count`` and ``epochs`` must be specified.


        Calibration is highly dependent on the base rate of positive triples. Therefore, for mode (2) of operation,
        the user is required to provide the ``positive_base_rate`` argument. For mode (1), that can be inferred
        automatically by the relative sizes of the positive and negative sets, but the user can override that by
        providing a value to ``positive_base_rate``.

        When calibrating without negatives, the most difficult problem is defining the positive base rate. This is
        determined by the user's selection of which triples will be analyzed during testing.

        Take WN11 as an example: it has approximately 50% positive triples on both the validation and test sets,
        so the positive base rate is 50%. However, if the user resamples it to have 75% positives and 25% negatives,
        the prior calibration will be impaired. The user must now recalibrate the model with a 75% positive base
        rate. As a result, this parameter is defined by how the user interacts with the dataset and cannot be
        determined mechanically or a priori.

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

        Examples:

        >>> import numpy as np
        >>> from sklearn.metrics import brier_score_loss, log_loss
        >>> from scipy.special import expit
        >>>
        >>> from emgraph.datasets import BaseDataset, DatasetType
        >>> from emgraph.models import ComplEx
        >>>
        >>> X = BaseDataset.load_dataset(DatasetType.WN11)
        >>> X_valid_pos = X['valid'][X['valid_labels']]
        >>> X_valid_neg = X['valid'][~X['valid_labels']]
        >>>
        >>> model = ComplEx(batches_count=64, seed=0, epochs=500, k=100, eta=20,
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
        """Predicts probabilities using the Platt scaling model (after calibration).

        Model must be calibrated beforehand with the ``calibrate`` method.

        :param X: Numpy array of triples to be evaluated.
        :type X: ndarray, shape [n, 3]
        :return: Probability of each triple to be true according to the Platt scaling calibration.
        :rtype: ndarray, shape [n, 3]
        """
        __doc__ = super()._calibrate.__doc__  # NOQA
        return super()._predict_proba(X)
