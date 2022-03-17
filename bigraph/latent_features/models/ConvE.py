

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

    def _get_model_loss(self, dataset_iterator):
        """Get the current loss including loss due to regularization.
        This function must be overridden if the model uses combination of different losses(eg: VAE).

        :param dataset_iterator: Dataset iterator.
        :type dataset_iterator: tf.data.Iterator
        :return: The loss value that must be minimized.
        :rtype: tf.Tensor
        """

        # training input placeholder
        self.x_pos_tf, self.y_true = dataset_iterator.get_next()

        # list of dependent ops that need to be evaluated before computing the loss
        dependencies = []

        # run the dependencies
        with tf.control_dependencies(dependencies):

            # look up embeddings from input training triples
            e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(self.x_pos_tf)

            # Get positive predictions
            self.y_pred = self._fn(e_s_pos, e_p_pos, e_o_pos)

            # Label smoothing and/or weighting is applied within Loss class
            loss = self.loss.apply(self.y_true, self.y_pred)

            if self.regularizer is not None:
                loss += self.regularizer.apply([self.ent_emb, self.rel_emb])

            return loss

    def _save_trained_params(self):
        """After model fitting, save all the trained parameters in trained_model_params in some order.
        The order would be useful for loading the model.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        """

        params_dict = {}
        params_dict['ent_emb'] = self.sess_train.run(self.ent_emb)
        params_dict['rel_emb'] = self.sess_train.run(self.rel_emb)
        params_dict['conv2d_W'] = self.sess_train.run(self.conv2d_W)
        params_dict['conv2d_B'] = self.sess_train.run(self.conv2d_B)
        params_dict['dense_W'] = self.sess_train.run(self.dense_W)
        params_dict['dense_B'] = self.sess_train.run(self.dense_B)

        if self.embedding_model_params['use_batchnorm']:

            bn_dict = {}

            for scope in ['batchnorm_input', 'batchnorm_conv', 'batchnorm_dense']:

                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                variables = [x for x in variables if 'Adam' not in x.name]  # Filter out any Adam variables

                var_dict = {x.name.split('/')[-1].split(':')[0]: x for x in variables}
                bn_dict[scope] = {'beta': self.sess_train.run(var_dict['beta']),
                                  'gamma': self.sess_train.run(var_dict['gamma']),
                                  'moving_mean': self.sess_train.run(var_dict['moving_mean']),
                                  'moving_variance': self.sess_train.run(var_dict['moving_variance'])}

            params_dict['bn_vars'] = bn_dict

        if self.embedding_model_params['use_bias']:
            params_dict['bias'] = self.sess_train.run(self.bias)

        params_dict['output_mapping'] = self.output_mapping

        self.trained_model_params = params_dict

    def _load_model_from_trained_params(self):
        """Load the model from trained params.
            While restoring make sure that the order of loaded parameters match the saved order.
            It's the duty of the embedding model to load the variables correctly.
            This method must be overridden if the model has any other parameters (apart from entity-relation embeddings)
            This function also set's the evaluation mode to do lazy loading of variables based on the number of
            distinct entities present in the graph.
        """

        # Generate the batch size based on entity length and batch_count
        self.batch_size = int(np.ceil(len(self.ent_to_idx) / self.batches_count))

        with tf.variable_scope('meta'):
            self.tf_is_training = tf.Variable(False, trainable=False)
            self.set_training_true = tf.assign(self.tf_is_training, True)
            self.set_training_false = tf.assign(self.tf_is_training, False)

        self.ent_emb = tf.Variable(self.trained_model_params['ent_emb'], dtype=tf.float32)
        self.rel_emb = tf.Variable(self.trained_model_params['rel_emb'], dtype=tf.float32)

        self.conv2d_W = tf.Variable(self.trained_model_params['conv2d_W'], dtype=tf.float32)
        self.conv2d_B = tf.Variable(self.trained_model_params['conv2d_B'], dtype=tf.float32)
        self.dense_W = tf.Variable(self.trained_model_params['dense_W'], dtype=tf.float32)
        self.dense_B = tf.Variable(self.trained_model_params['dense_B'], dtype=tf.float32)

        if self.embedding_model_params['use_batchnorm']:
            self.bn_vars = self.trained_model_params['bn_vars']

        if self.embedding_model_params['use_bias']:
            self.bias = tf.Variable(self.trained_model_params['bias'], dtype=tf.float32)

        self.output_mapping = self.trained_model_params['output_mapping']

    def _fn(self, e_s, e_p, e_o):
        r"""The ConvE scoring function.

        The function implements the scoring function as defined by
        .. math::

            f(vec(f([\overline{e_s};\overline{r_r}] * \Omega)) W ) e_o

        Additional details for equivalence of the models available in :cite:`Dettmers2016`.

        :param e_s: The embeddings of a list of subjects.
        :type e_s: tf.Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: tf.Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: tf.Tensor, shape [n]
        :return: The operation corresponding to the scoring function.
        :rtype: tf.Op
        """

        def _dropout(X, rate):
            dropout_rate = tf.cond(self.tf_is_training, true_fn=lambda: tf.constant(rate),
                                   false_fn=lambda: tf.constant(0, dtype=tf.float32))
            out = tf.nn.dropout(X, rate=dropout_rate)
            return out

        def _batchnorm(X, key, axis):

            with tf.variable_scope(key, reuse=tf.AUTO_REUSE):
                x = tf.compat.v1.layers.batch_normalization(X, training=self.tf_is_training, axis=axis,
                                                            beta_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['beta']),
                                                            gamma_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['gamma']),
                                                            moving_mean_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['moving_mean']),
                                                            moving_variance_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['moving_variance']))
            return x

        # Inputs
        stacked_emb = tf.stack([e_s, e_p], axis=2)
        self.inputs = tf.reshape(stacked_emb,
                                 shape=[tf.shape(stacked_emb)[0], self.embedding_model_params['embed_image_height'],
                                        self.embedding_model_params['embed_image_width'], 1])

        x = self.inputs

        if self.embedding_model_params['use_batchnorm']:
            x = _batchnorm(x, key='batchnorm_input', axis=3)

        if not self.embedding_model_params['dropout_embed'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_embed'])

        # Convolution layer
        x = tf.nn.conv2d(x, self.conv2d_W, [1, 1, 1, 1], padding='VALID')

        if self.embedding_model_params['use_batchnorm']:
            x = _batchnorm(x, key='batchnorm_conv', axis=3)
        else:
            # Batch normalization will cancel out bias, so only add bias term if not using batchnorm
            x = tf.nn.bias_add(x, self.conv2d_B)

        x = tf.nn.relu(x)

        if not self.embedding_model_params['dropout_conv'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_conv'])

        # Dense layer
        x = tf.reshape(x, shape=[tf.shape(x)[0], self.embedding_model_params['dense_dim']])
        x = tf.matmul(x, self.dense_W)

        if self.embedding_model_params['use_batchnorm']:
            # Initializing batchnorm vars for dense layer with shape=[1] will still broadcast over the shape of
            # the specified axis, e.g. dense shape = [?, k], batchnorm on axis 1 will create k batchnorm vars.
            # This is layer normalization rather than batch normalization, so adding a dimension to keep batchnorm,
            # thus dense shape = [?, k, 1], batchnorm on axis 2.
            x = tf.expand_dims(x, -1)
            x = _batchnorm(x, key='batchnorm_dense', axis=2)
            x = tf.squeeze(x, -1)
        else:
            x = tf.nn.bias_add(x, self.dense_B)

        # Note: Reference ConvE implementation had dropout on dense layer before applying batch normalization.
        # This can cause variance shift and reduce model performance, so have moved it after as recommended in:
        # https://arxiv.org/abs/1801.05134
        if not self.embedding_model_params['dropout_dense'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_dense'])

        x = tf.nn.relu(x)
        x = tf.matmul(x, tf.transpose(self.ent_emb))

        if self.embedding_model_params['use_bias']:
            x = tf.add(x, self.bias)

        self.scores = x

        return self.scores