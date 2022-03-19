
import numpy as np
import tensorflow as tf
import logging

from .EmbeddingModel import EmbeddingModel, register_model, ENTITY_THRESHOLD
from ..initializers import DEFAULT_XAVIER_IS_UNIFORM
from bigraph.latent_features import constants as constants
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register_model("ConvKB", {'num_filters': 32, 'filter_sizes': [1], 'dropout': 0.1})
class ConvKB(EmbeddingModel):
    r"""Convolution-based model

    The ConvKB model :cite:`Nguyen2018`:

    .. math::

        f_{ConvKB}= concat \,(g \, ([\mathbf{e}_s, \mathbf{r}_p, \mathbf{e}_o]) * \Omega)) \cdot W

    where :math:`g` is a non-linear function,  :math:`*` is the convolution operator,
    :math:`\cdot` is the dot product, :math:`concat` is the concatenation operator
    and :math:`\Omega` is a set of filters.

    .. note::
        The evaluation protocol implemented in :meth:`ampligraph.evaluation.evaluate_performance` assigns the worst rank
        to a positive test triple in case of a tie with negatives. This is the agreed upon behaviour in literature.
        The original ConvKB implementation :cite:`Nguyen2018` assigns instead the top rank, hence leading to
        `results which are not directly comparable with
        literature <https://github.com/daiquocnguyen/ConvKB/issues/5>`_ .
        We report results obtained with the agreed-upon protocol (tie=worst rank). Note that under these conditions
        the model :ref:`does not reach the state-of-the-art results claimed in the original paper<eval_experiments>`.

    Examples:

    >>> from bigraph.latent_features import ConvKB
    >>> from bigraph.datasets import load_wn18
    >>> model = ConvKB(batches_count=2, seed=22, epochs=1, k=10, eta=1,
    >>>               embedding_model_params={'num_filters': 32, 'filter_sizes': [1],
    >>>                                       'dropout': 0.1},
    >>>               optimizer='adam', optimizer_params={'lr': 0.001},
    >>>               loss='pairwise', loss_params={}, verbose=True)
    >>>
    >>> X = load_wn18()
    >>>
    >>> model.fit(X['train'])
    >>>
    >>> print(model.predict(X['test'][:5]))
    [[0.2803744], [0.0866661], [0.012815937], [-0.004235901], [-0.010947697]]
    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'num_filters': 32,
                                         'filter_sizes': [1],
                                         'dropout': 0.1},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 large_graphs=False,
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize ConvKB class.

        :param k: Embedding space dimensionality.
        :type k: int
        :param eta: The number of negatives that must be generated at runtime during training for each positive.
        :type eta: int
        :param epochs: The iterations of the training loop.
        :type epochs: int
        :param batches_count: The number of batches in which the training set must be split during the training loop.
        :type batches_count: int
        :param seed: The seed used by the internal random numbers generator.
        :type seed: int
        :param embedding_model_params: ConvKB-specific hyperparams:
            - **num_filters** - Number of feature maps per convolution kernel. Default: 32
            - **filter_sizes** - List of convolution kernel sizes. Default: [1]
            - **dropout** - Dropout on the embedding layer. Default: 0.0
            - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
            - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value
            until it reachs original value.
            - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph
            topology.
            - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]

            The last 4 parameters are related to FocusE layers.
        :type embedding_model_params: dict
        :param optimizer: The optimizer used to minimize the loss function. Choose between
            'sgd', 'adagrad', 'adam', 'momentum'.
        :type optimizer: str
        :param optimizer_params: Arguments specific to the optimizer, passed as a dictionary.

        Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

        Example: ``optimizer_params={'lr': 0.01}``
        :type optimizer_params:
        :param loss: The type of loss function to use during training.
        :type loss: str
        :param loss_params: Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>`
            documentation for additional details.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01, 'label_smoothing': 0.1}``
        :type loss_params: dict
        :param regularizer: The regularization strategy to use with the loss function.

            - ``None``: the model will not use any regularizer (default)
            - ``LP``: the model will use L1, L2 or L3 based on the value of ``regularizer_params['p']`` (see below).
        :type regularizer: str
        :param regularizer_params: Dictionary of regularizer-specific hyperparameters. See the
            :ref:`regularizers <ref-reg>`
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
        :type initializer_params: str
        :param large_graphs: Avoid loading entire dataset onto GPU when dealing with large graphs.
        :type large_graphs: bool
        :param verbose: Verbose mode.
        :type verbose: bool
        """

        num_filters = embedding_model_params['num_filters']
        filter_sizes = embedding_model_params['filter_sizes']

        if isinstance(filter_sizes, int):
            filter_sizes = [filter_sizes]

        dense_dim = (k * len(filter_sizes) - sum(filter_sizes) + len(filter_sizes)) * num_filters
        embedding_model_params['dense_dim'] = dense_dim
        embedding_model_params['filter_sizes'] = filter_sizes

        super().__init__(k=k, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         large_graphs=large_graphs, verbose=verbose)