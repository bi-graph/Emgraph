

import numpy as np
import tensorflow as tf
import logging
from sklearn.utils import check_random_state
from tqdm import tqdm
from functools import partial
import time

from .EmbeddingModel import EmbeddingModel, register_model, ENTITY_THRESHOLD
from ..initializers import DEFAULT_XAVIER_IS_UNIFORM
from bigraph.latent_features import constants as constants

from ...datasets import OneToNDatasetAdapter
from ..optimizers import SGDOptimizer
from ...evaluation import to_idx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register_model('ConvE', ['conv_filters', 'conv_kernel_size', 'dropout_embed', 'dropout_conv',
                          'dropout_dense', 'use_bias', 'use_batchnorm'], {})
class ConvE(EmbeddingModel):
    r""" Convolutional 2D KG Embeddings

    The ConvE model :cite:`DettmersMS018`.

    ConvE uses convolutional layers.
    :math:`g` is a non-linear activation function, :math:`\ast` is the linear convolution operator,
    :math:`vec` indicates 2D reshaping.

    .. math::

        f_{ConvE} =  \langle \sigma \, (vec \, ( g \, ([ \overline{\mathbf{e}_s} ; \overline{\mathbf{r}_p} ]
        \ast \Omega )) \, \mathbf{W} )) \, \mathbf{e}_o\rangle


    .. note::

        ConvE does not handle 's+o' corruptions currently, nor ``large_graph`` mode.


    Examples:

    >>> import numpy as np
    >>> from bigraph.latent_features import ConvE
    >>> model = ConvE(batches_count=1, seed=22, epochs=5, k=100)
    >>>
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
    [0.42921206 0.38998795]
    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'conv_filters': constants.DEFAULT_CONVE_CONV_FILTERS,
                                         'conv_kernel_size': constants.DEFAULT_CONVE_KERNEL_SIZE,
                                         'dropout_embed': constants.DEFAULT_CONVE_DROPOUT_EMBED,
                                         'dropout_conv': constants.DEFAULT_CONVE_DROPOUT_CONV,
                                         'dropout_dense': constants.DEFAULT_CONVE_DROPOUT_DENSE,
                                         'use_bias': constants.DEFAULT_CONVE_USE_BIAS,
                                         'use_batchnorm': constants.DEFAULT_CONVE_USE_BATCHNORM},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss='bce',
                 loss_params={'label_weighting': False,
                              'label_smoothing': 0.1},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 low_memory=False,
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize a ConvE model

        Also creates a new Tensorflow session for training.

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
        :param embedding_model_params: ConvE-specific hyperparams:

            - **conv_filters** (int): Number of convolution feature maps. Default: 32
            - **conv_kernel_size** (int): Convolution kernel size. Default: 3
            - **dropout_embed** (float|None): Dropout on the embedding layer. Default: 0.2
            - **dropout_conv** (float|None): Dropout on the convolution maps. Default: 0.3
            - **dropout_dense** (float|None): Dropout on the dense layer. Default: 0.2
            - **use_bias** (bool): Use bias layer. Default: True
            - **use_batchnorm** (bool): Use batch normalization after input, convolution, dense layers. Default: True
        :type embedding_model_params: dict
        :param optimizer: The optimizer used to minimize the loss function. Choose between
            'sgd', 'adagrad', 'adam', 'momentum'.
        :type optimizer: str
        :param optimizer_params: Arguments specific to the optimizer, passed as a dictionary.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01}``
        :type optimizer_params: dict
        :param loss: The type of loss function to use during training.

            - ``bce``  the model will use binary cross entropy loss function.
        :type loss: str
        :param loss_params: Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>` documentation for
            additional details.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.
            - **'label_smoothing'** (float): applies label smoothing to one-hot outputs. Default: 0.1.
            - **'label_weighting'** (bool): applies label weighting to one-hot outputs. Default: True

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
        :type initializer_params: dict
        :param low_memory: Train ConvE with a (slower) low_memory option. If MemoryError is still encountered, try raising the
            batches_count value. Default: False.
        :type low_memory: bool
        :param verbose: Verbose mode.
        :type verbose: bool
        """

        # Add default values if not provided in embedding_model_params dict
        default_embedding_model_params = {'conv_filters': constants.DEFAULT_CONVE_CONV_FILTERS,
                                          'conv_kernel_size': constants.DEFAULT_CONVE_KERNEL_SIZE,
                                          'dropout_embed': constants.DEFAULT_CONVE_DROPOUT_EMBED,
                                          'dropout_conv': constants.DEFAULT_CONVE_DROPOUT_CONV,
                                          'dropout_dense': constants.DEFAULT_CONVE_DROPOUT_DENSE,
                                          'use_batchnorm': constants.DEFAULT_CONVE_USE_BATCHNORM,
                                          'use_bias': constants.DEFAULT_CONVE_USE_BATCHNORM}

        for key, val in default_embedding_model_params.items():
            if key not in embedding_model_params.keys():
                embedding_model_params[key] = val

        # Find factor pairs (i,j) of concatenated embedding dimensions, where min(i,j) >= conv_kernel_size
        n = k * 2
        emb_img_depth = 1

        ksize = embedding_model_params['conv_kernel_size']
        nfilters = embedding_model_params['conv_filters']

        emb_img_width, emb_img_height = None, None
        for i in range(int(np.sqrt(n)) + 1, ksize, -1):
            if n % i == 0:
                emb_img_width, emb_img_height = (i, int(n / i))
                break

        if not emb_img_width and not emb_img_height:
            msg = 'Unable to determine factor pairs for embedding reshape. Choose a smaller convolution kernel size, ' \
                  'or a larger embedding dimension.'
            logger.info(msg)
            raise ValueError(msg)

        embedding_model_params['embed_image_width'] = emb_img_width
        embedding_model_params['embed_image_height'] = emb_img_height
        embedding_model_params['embed_image_depth'] = emb_img_depth

        # Calculate dense dimension
        embedding_model_params['dense_dim'] = (emb_img_width - (ksize - 1)) * (emb_img_height - (ksize - 1)) * nfilters

        self.low_memory = low_memory

        super().__init__(k=k, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         verbose=verbose)

    def _initialize_parameters(self):
        """Initialize parameters of the model.

            This function creates and initializes entity and relation embeddings (with size k).
            If the graph is large, then it loads only the required entity embeddings (max:batch_size*2)
            and all relation embeddings.
            Override this function if the parameters needs to be initialized differently.
        """
        timestamp = int(time.time() * 1e6)
        if not self.dealing_with_large_graphs:

            with tf.variable_scope('meta'):
                self.tf_is_training = tf.Variable(False, trainable=False)
                self.set_training_true = tf.assign(self.tf_is_training, True)
                self.set_training_false = tf.assign(self.tf_is_training, False)

            nfilters = self.embedding_model_params['conv_filters']
            ninput = self.embedding_model_params['embed_image_depth']
            ksize = self.embedding_model_params['conv_kernel_size']
            dense_dim = self.embedding_model_params['dense_dim']

            self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                           shape=[len(self.ent_to_idx), self.k],
                                           initializer=self.initializer.get_entity_initializer(
                                               len(self.ent_to_idx), self.k),
                                           dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                           shape=[len(self.rel_to_idx), self.k],
                                           initializer=self.initializer.get_relation_initializer(
                                               len(self.rel_to_idx), self.k),
                                           dtype=tf.float32)

            self.conv2d_W = tf.get_variable('conv2d_weights_{}'.format(timestamp),
                                            shape=[ksize, ksize, ninput, nfilters],
                                            initializer=tf.initializers.he_normal(seed=self.seed),
                                            dtype=tf.float32)
            self.conv2d_B = tf.get_variable('conv2d_bias_{}'.format(timestamp),
                                            shape=[nfilters],
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.dense_W = tf.get_variable('dense_weights_{}'.format(timestamp),
                                           shape=[dense_dim, self.k],
                                           initializer=tf.initializers.he_normal(seed=self.seed),
                                           dtype=tf.float32)
            self.dense_B = tf.get_variable('dense_bias_{}'.format(timestamp),
                                           shape=[self.k],
                                           initializer=tf.zeros_initializer(), dtype=tf.float32)

            if self.embedding_model_params['use_batchnorm']:
                emb_img_dim = self.embedding_model_params['embed_image_depth']

                self.bn_vars = {'batchnorm_input': {'beta': np.zeros(shape=[emb_img_dim]),
                                                    'gamma': np.ones(shape=[emb_img_dim]),
                                                    'moving_mean': np.zeros(shape=[emb_img_dim]),
                                                    'moving_variance': np.ones(shape=[emb_img_dim])},
                                'batchnorm_conv': {'beta': np.zeros(shape=[nfilters]),
                                                   'gamma': np.ones(shape=[nfilters]),
                                                   'moving_mean': np.zeros(shape=[nfilters]),
                                                   'moving_variance': np.ones(shape=[nfilters])},
                                'batchnorm_dense': {'beta': np.zeros(shape=[1]),  # shape = [1] for batch norm
                                                    'gamma': np.ones(shape=[1]),
                                                    'moving_mean': np.zeros(shape=[1]),
                                                    'moving_variance': np.ones(shape=[1])}}

            if self.embedding_model_params['use_bias']:
                self.bias = tf.get_variable('activation_bias_{}'.format(timestamp),
                                            shape=[1, len(self.ent_to_idx)],
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs.')