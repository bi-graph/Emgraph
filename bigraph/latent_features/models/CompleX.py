
from .EmbeddingModel import EmbeddingModel, register_model
from bigraph.latent_features import constants as constants
from bigraph.latent_features.initializers import DEFAULT_XAVIER_IS_UNIFORM
import tensorflow as tf
import time


@register_model("ComplEx", ["negative_corruption_entities"])
class ComplEx(EmbeddingModel):
    r"""Complex embeddings (ComplEx)

    The ComplEx model :cite:`trouillon2016complex` is an extension of
    the :class:`ampligraph.latent_features.DistMult` bilinear diagonal model
    . ComplEx scoring function is based on the trilinear Hermitian dot product in :math:`\mathcal{C}`:

    .. math::

        f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \rangle)

    ComplEx can be improved if used alongside the nuclear 3-norm
    (the **ComplEx-N3** model :cite:`lacroix2018canonical`), which can be easily added to the
    loss function via the ``regularizer`` hyperparameter with ``p=3`` and
    a chosen regularisation weight (represented by ``lambda``), as shown in the example below.
    See also :meth:`ampligraph.latent_features.LPRegularizer`.

    .. note::

        Since ComplEx embeddings belong to :math:`\mathcal{C}`, this model uses twice as many parameters as
        :class:`ampligraph.latent_features.DistMult`.

    Examples
    --------
    >>> import numpy as np
    >>> from bigraph.latent_features import ComplEx
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

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'negative_corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
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

        self.internal_k = self.k * 2

    def _initialize_parameters(self):
        """Initialize the complex embeddings.

        This function creates and initializes entity and relation embeddings (with size k).
        If the graph is large, then it loads only the required entity embeddings (max:batch_size*2)
        and all relation embeddings.
        """
        timestamp = int(time.time() * 1e6)
        if not self.dealing_with_large_graphs:
            self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                           shape=[len(self.ent_to_idx), self.internal_k],
                                           initializer=self.initializer.get_entity_initializer(
                                           len(self.ent_to_idx), self.internal_k),
                                           dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                           shape=[len(self.rel_to_idx), self.internal_k],
                                           initializer=self.initializer.get_relation_initializer(
                                           len(self.rel_to_idx), self.internal_k),
                                           dtype=tf.float32)
        else:
            # initialize entity embeddings to zero (these are reinitialized every batch by batch embeddings)
            self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                           shape=[self.batch_size * 2, self.internal_k],
                                           initializer=tf.zeros_initializer(),
                                           dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                           shape=[len(self.rel_to_idx), self.internal_k],
                                           initializer=self.initializer.get_relation_initializer(
                                           len(self.rel_to_idx), self.internal_k),
                                           dtype=tf.float32)

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
        return tf.reduce_sum(e_p_real * e_s_real * e_o_real, axis=1) + \
            tf.reduce_sum(e_p_real * e_s_img * e_o_img, axis=1) + \
            tf.reduce_sum(e_p_img * e_s_real * e_o_img, axis=1) - \
            tf.reduce_sum(e_p_img * e_s_img * e_o_real, axis=1)