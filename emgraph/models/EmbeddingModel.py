# todo: rename most of these functions' names
import abc
import functools
import logging
import time
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from tqdm import tqdm

from emgraph.datasets import EmgraphBaseDatasetAdaptor, NumpyDatasetAdapter
from emgraph.evaluation import (
    generate_corruptions_for_eval,
    generate_corruptions_for_fit,
    hits_at_n_score,
    mrr_score,
    to_idx,
)
from emgraph.initializers._initializer_constants import (
    DEFAULT_GLOROT_IS_UNIFORM,
    INITIALIZER_REGISTRY,
)
from emgraph.losses._loss_constants import LOSS_REGISTRY
from emgraph.regularizers._regularizer_constants import REGULARIZER_REGISTRY
from emgraph.training._optimizer_constants import OPTIMIZER_REGISTRY
from emgraph.training.sgd import SGD
from emgraph.utils import constants as constants
from emgraph.utils.misc import make_variable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_REGISTRY = {}

ENTITY_THRESHOLD = 5e5

tf.device("/physical_device:GPU:0")  # todo: fix me


def set_entity_threshold(threshold):
    """Sets the entity threshold (threshold after which large graph mode is initiated)

    :param threshold: Threshold for a graph to be considered as a big graph
    :type threshold: int
    :return:
    :rtype:
    """

    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = threshold


def reset_entity_threshold():
    """Resets the entity threshold"""
    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = 5e5


def register_model(name, external_params=None, class_params=None):
    """Wrapper for Saving the class info in the MODEL_REGISTRY dictionary.

    :param name: Name of the class
    :type name: str
    :param external_params: External parameters
    :type external_params: list
    :param class_params: Class parameters
    :type class_params: dict
    :return: Class object
    :rtype: object
    """
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        MODEL_REGISTRY[name] = class_handle
        class_handle.name = name
        MODEL_REGISTRY[name].external_params = external_params
        MODEL_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


# todo: rename this
@tf.custom_gradient
def custom_softplus(x):
    e = 9999 * tf.exp(x)

    def grad(dy):
        return dy * (1 - 1 / (1 + e))

    return tf.math.log(1 + e), grad


class EmbeddingModel(abc.ABC):
    """Abstract class for embedding models

    Emgraph neural knowledge graph embeddings models extend this class and its core methods.

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
    :param embedding_model_params: Model-specific hyperparams, passed to the model as a dictionary.
    Refer to model-specific documentation for details.

    For FocusE Layer, following hyper-params can be passed:

        - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
        - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value
        until it reaches original value.
        - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph
        topology.
    - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]
    :type embedding_model_params: dict
    :param optimizer: The optimizer used to minimize the loss function. Choose between
        'sgd', 'adagrad', 'adam', 'momentum'.
    :type optimizer: str
    :param optimizer_params: Arguments specific to the optimizer, passed as a dictionary.

        Supported keys:

        - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
        - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

        Example: ``optimizer_params={'lr': 0.009}``
    :type optimizer_params: dict
    :param loss: The type of loss function to use during training.

        - ``pairwise``  the model will use pairwise margin-based loss function.
        - ``nll`` the model will use negative loss likelihood.
        - ``absolute_margin`` the model will use absolute margin likelihood.
        - ``self_adversarial`` the model will use adversarial sampling loss function.
        - ``multiclass_nll`` the model will use multiclass nll loss. Switch to multiclass loss defined in
          :cite:`chen2015` by passing 'corrupt_side' as ['s','o'] to embedding_model_params.
          To use loss defined in :cite:`kadlecBK17` pass 'corrupt_side' as 'o' to embedding_model_params.
    :type loss: str
    :param loss_params: Dictionary of loss-specific hyperparameters. See :ref:`loss
        functions <loss>`
        documentation for additional details.

        Example: ``optimizer_params={'lr': 0.01}`` if ``loss='pairwise'``.
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
    :param large_graphs: Avoid loading entire dataset onto GPU when dealing with large graphs.
    :type large_graphs: bool
    :param verbose: Verbose mode.
    :type verbose: bool

    """

    def __init__(
        self,
        k=constants.DEFAULT_EMBEDDING_SIZE,
        eta=constants.DEFAULT_ETA,
        epochs=constants.DEFAULT_EPOCH,
        batches_count=constants.DEFAULT_BATCH_COUNT,
        seed=constants.DEFAULT_SEED,
        embedding_model_params={},
        optimizer=constants.DEFAULT_OPTIM,
        optimizer_params={"lr": constants.DEFAULT_LR},
        loss=constants.DEFAULT_LOSS,
        loss_params={},
        regularizer=constants.DEFAULT_REGULARIZER,
        regularizer_params={},
        initializer=constants.DEFAULT_INITIALIZER,
        initializer_params={"uniform": DEFAULT_GLOROT_IS_UNIFORM},
        large_graphs=False,
        verbose=constants.DEFAULT_VERBOSE,
        model_variables=None,
    ):
        """Initialize the EmbeddingModel class"""

        if (loss == "bce") ^ (self.name == "ConvE"):
            raise ValueError(
                "Invalid Model - Loss combination. "
                "ConvE model can be used with BCE loss only and vice versa."
            )

        # Store for restoring later.
        self.all_params = {
            "k": k,
            "eta": eta,
            "epochs": epochs,
            "batches_count": batches_count,
            "seed": seed,
            "embedding_model_params": embedding_model_params,
            "optimizer": optimizer,
            "optimizer_params": optimizer_params,
            "loss": loss,
            "loss_params": loss_params,
            "regularizer": regularizer,
            "regularizer_params": regularizer_params,
            "initializer": initializer,
            "initializer_params": initializer_params,
            "verbose": verbose,
        }
        # tf.reset_default_graph()
        self.seed = seed
        self.rnd = check_random_state(self.seed)
        # tf.random.set_random_seed(seed)
        tf.random.set_seed(seed)

        self.is_filtered = False
        self.use_focusE = False
        self.loss_params = loss_params

        self.embedding_model_params = embedding_model_params

        self.k = k
        self.internal_k = k
        self.epochs = epochs
        self.eta = eta
        self.regularizer_params = regularizer_params
        self.batches_count = batches_count

        self.dealing_with_large_graphs = large_graphs

        if batches_count == 1:
            logger.warning(
                "All triples will be processed in the same batch (batches_count=1). "
                "When processing large graphs it is recommended to batch the input knowledge graph instead."
            )

        try:
            self.loss = LOSS_REGISTRY[loss](self.eta, self.loss_params, verbose=verbose)
        except KeyError:
            msg = "Unsupported loss function: {}".format(loss)
            logger.error(msg)
            raise ValueError(msg)

        try:
            if regularizer is not None:
                self.regularizer = REGULARIZER_REGISTRY[regularizer](
                    self.regularizer_params, verbose=verbose
                )
            else:
                self.regularizer = regularizer
        except KeyError:
            msg = "Unsupported regularizer: {}".format(regularizer)
            logger.error(msg)
            raise ValueError(msg)

        self.optimizer_params = optimizer_params

        try:
            self.optimizer = OPTIMIZER_REGISTRY[optimizer](
                self.optimizer_params, self.batches_count, verbose
            )
        except KeyError:
            msg = "Unsupported optimizer: {}".format(optimizer)
            logger.error(msg)
            raise ValueError(msg)

        self.verbose = verbose

        self.initializer_params = initializer_params

        try:
            self.initializer = INITIALIZER_REGISTRY[initializer](
                self.initializer_params, verbose, self.rnd
            )
        except KeyError:
            msg = "Unsupported initializer: {}".format(initializer)
            logger.error(msg)
            raise ValueError(msg)

        # self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config = tf.config
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        try:
            self.tf_config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
        self.sess_train = None  # todo: remove its usage
        self.trained_model_params = []
        self.is_fitted = False
        self.eval_config = {}
        self.eval_dataset_handle = None
        self.train_dataset_handle = None
        self.is_calibrated = False
        self.calibration_parameters = []

    @abc.abstractmethod
    def _fn(self, e_s, e_p, e_o):
        # todo: rename this to _calculate_score
        """The scoring function of the model.

        A model-specific strategy is used to assign a score to a list of triples. Triples are passed in the form of
        lists of subject, predicate, and object embeddings. Every model must override this function in order to
        return the corresponding score.

        :param e_s: The embeddings of a list of subjects.
        :type e_s: tf.Tensor, shape [n]
        :param e_p: The embeddings of a list of predicates.
        :type e_p: tf.Tensor, shape [n]
        :param e_o: The embeddings of a list of objects.
        :type e_o: tf.Tensor, shape [n]
        :return: The operation corresponding to the scoring function.
        :rtype: tf.Op
        """

        logger.error("_fn is a placeholder function in an abstract class")
        NotImplementedError("This function is a placeholder in an abstract class")

    def get_hyperparameter_dict(self):
        """Return the hyperparameters of the model.

        :return: Dictionary of hyperparameters that were used for training.
        :rtype: dict
        """

        return self.all_params

    def get_embedding_model_params(self, output_dict):
        """Save the model parameters in the dictionary.

        :param output_dict: Saved parameters dictionary. The model saves the parameters, and it can be restored later
        :type output_dict: dict
        :return:
        :rtype:
        """

        output_dict["model_params"] = self.trained_model_params
        output_dict["large_graph"] = self.dealing_with_large_graphs
        output_dict["calibration_parameters"] = self.calibration_parameters

    def restore_model_params(self, in_dict):
        """Load the model parameters from the input dictionary.

        :param in_dict: Saved parameters dictionary. The model loads the parameters.
        :type in_dict: dict
        :return:
        :rtype:
        """

        self.trained_model_params = in_dict["model_params"]

        # Try catch is for backward compatibility
        try:
            self.calibration_parameters = in_dict["calibration_parameters"]
        except KeyError:
            # For backward compatibility
            self.calibration_parameters = []

        # Try catch is for backward compatibility
        try:
            self.dealing_with_large_graphs = in_dict["large_graph"]
        except KeyError:
            # For backward compatibility
            self.dealing_with_large_graphs = False

    def _save_trained_params(self):
        """
        After training the model, save all parameters in some order in trained model params. When loading the model,
        the order is used. If the model has any extra parameters, this method must be overridden (apart from
        entity-relation embeddings).
        """
        params_to_save = []
        if not self.dealing_with_large_graphs:
            # params_to_save.append(self.sess_train.run(self.ent_emb))
            params_to_save.append(self.ent_emb)
        else:
            params_to_save.append(self.ent_emb_cpu)

        # params_to_save.append(self.sess_train.run(self.rel_emb))
        params_to_save.append(self.rel_emb)

        self.trained_model_params = params_to_save

    def _load_model_from_trained_params(self):
        """
                Load the model using the trained parameters.

        Make sure that the order of the loaded parameters matches the saved order when restoring. The embedding model
        is responsible for accurately loading the variables. If the model has any extra parameters, this method must
        be overridden (apart from entity-relation embeddings). This function additionally configures the evaluation
        mode to do lazy variable loading dependent on the amount of variables. The graph contains separate items.
        """

        # Generate the batch size based on entity length and batch_count
        self.batch_size = int(np.ceil(len(self.ent_to_idx) / self.batches_count))

        if len(self.ent_to_idx) > ENTITY_THRESHOLD:
            self.dealing_with_large_graphs = True

            logger.warning(
                "Your graph has a large number of distinct entities. "
                "Found {} distinct entities".format(len(self.ent_to_idx))
            )

            logger.warning(
                "Changing the variable loading strategy to use lazy loading of variables..."
            )
            logger.warning("Evaluation would take longer than usual.")

        if not self.dealing_with_large_graphs:
            # (We use tf.variable for future - to load and continue training)
            self.ent_emb = tf.Variable(self.trained_model_params[0], dtype=tf.float32)
        else:
            # All of the corrupted entities' embeddings will not fit on the GPU. We loaded batch size*2 embeddings on
            # GPU during training since only 2* batch size unique entities can be present in one batch.
            #
            # During corruption generation in eval mode, one side (s/o) remains constant while the other side
            # fluctuates. As a result, we employ a batch size of 2 * training batch size for corruption generation,
            # which means that those many corruption embeddings are loaded on the GPU each batch. In other words,
            # such corruptions would be dealt with as a group.

            self.corr_batch_size = self.batch_size * 2

            # Load the entity embeddings on the cpu
            self.ent_emb_cpu = self.trained_model_params[0]
            # (We use tf.variable for future - to load and continue training)
            # create empty variable on GPU.
            # we initialize it with zeros because the actual embeddings will be loaded on the fly.
            self.ent_emb = tf.Variable(
                np.zeros((self.corr_batch_size, self.internal_k)), dtype=tf.float32
            )

        # (We use tf.variable for future - to load and continue training)
        self.rel_emb = tf.Variable(self.trained_model_params[1], dtype=tf.float32)

    def get_embeddings(self, entities, embedding_type="entity"):
        """Get the embeddings of entities or relations.

        .. Note ::
            Use :meth:`emgraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.

        :param entities: The entities (or relations) of interest. Element of the vector must be the original string
        literals, and not internal IDs.
        :type entities: ndarray, shape=[n]
        :param embedding_type: If 'entity', ``entities`` argument will be considered as a list of knowledge graph entities (i.e. nodes).
            If set to 'relation', they will be treated as relation types instead (i.e. predicates).
        :type embedding_type: str
        :return: An array of k-dimensional embeddings.
        :rtype: ndarray, shape [n, k]
        """

        if not self.is_fitted:
            msg = "Model has not been fitted."
            logger.error(msg)
            raise RuntimeError(msg)

        if embedding_type == "entity":
            emb_list = self.trained_model_params[0]
            lookup_dict = self.ent_to_idx
        elif embedding_type == "relation":
            emb_list = self.trained_model_params[1]
            lookup_dict = self.rel_to_idx
        else:
            msg = "Invalid entity type: {}".format(embedding_type)
            logger.error(msg)
            raise ValueError(msg)

        idxs = np.vectorize(lookup_dict.get)(entities)
        return emb_list[idxs]

    def _lookup_embeddings(self, x, get_weight=False):
        """Get the embeddings for subjects, predicates, and objects of a list of statements used to train the model.

        :param x: A tensor of k-dimensional embeddings
        :type x: tf.Tensor, shape [n, k]
        :param get_weight: Flag indicates whether to return the weights
        :type get_weight: bool
        :return: e_s : A Tensor that includes the embeddings of the subjects.
        e_p : A Tensor that includes the embeddings of the predicates.
        e_o : A Tensor that includes the embeddings of the objects.
        :rtype: tf.Tensor, tf.Tensor, tf.Tensor
        """

        e_s = self._entity_lookup(x[:, 0])
        e_p = tf.nn.embedding_lookup(self.rel_emb, x[:, 1])
        e_o = self._entity_lookup(x[:, 2])

        if get_weight:
            wt = self.weight_triple[
                self.batch_number
                * self.batch_size : (self.batch_number + 1)
                * self.batch_size
            ]

            return e_s, e_p, e_o, wt
        return e_s, e_p, e_o

    def _entity_lookup(self, entity):
        """Get the embeddings for entities.
           Remaps the entity indices to corresponding variables in the GPU memory when dealing with large graphs.

        :param entity: Entity indices
        :type entity: tf.Tensor, shape [n, 1]
        :return: A Tensor that includes the embeddings of the entities.
        :rtype: tf.Tensor
        """

        if self.dealing_with_large_graphs:
            remapping = self.sparse_mappings.lookup(entity)
        else:
            remapping = entity

        emb = tf.nn.embedding_lookup(self.ent_emb, remapping)
        return emb

    # tf2x_dev branch
    def make_variable(
        self,
        name=None,
        shape=None,
        initializer=tf.keras.initializers.Zeros,
        dtype=tf.float32,
        trainable=True,
    ):
        return tf.Variable(
            initializer(shape=shape, dtype=dtype), name=name, trainable=trainable
        )

    def _initialize_parameters(self):
        """Initialize parameters of the model.

        This function is responsible for the creation and initialization of entity and relation embeddings (with size
        k). If the graph is huge, it only loads the entity embeddings that are required (max:batch size*2). as well
        as all relation embeddings.

        If the parameters must be initialized differently, override this function.
        """
        timestamp = int(time.time() * 1e6)
        if not self.dealing_with_large_graphs:
            print("shape: ", (len(self.ent_to_idx), self.internal_k))

            self.ent_emb = make_variable(
                name="ent_emb_{}".format(timestamp),
                shape=[len(self.ent_to_idx), self.internal_k],
                initializer=self.initializer.get_entity_initializer(),
                dtype=tf.float32,
            )

            self.rel_emb = make_variable(
                name="rel_emb_{}".format(timestamp),
                shape=[len(self.rel_to_idx), self.internal_k],
                initializer=self.initializer.get_relation_initializer(),
                dtype=tf.float32,
            )

            # self.ent_emb2 = tf.compat.v1.get_variable('ent_emb_{}'.format(timestamp),
            #                                           shape=[len(self.ent_to_idx), self.internal_k],
            #                                           initializer=self.initializer.get_entity_initializer(
            #                                               len(self.ent_to_idx), self.internal_k),
            #                                           dtype=tf.float32)

            # self.rel_emb = tf.compat.v1.get_variable('rel_emb_{}'.format(timestamp),
            #                                          shape=[len(self.rel_to_idx), self.internal_k],
            #                                          initializer=self.initializer.get_relation_initializer(
            #                                              len(self.rel_to_idx), self.internal_k),
            #                                          dtype=tf.float32)

        else:
            # initialize entity embeddings to zero (these are reinitialized every batch by batch embeddings)

            self.ent_emb = make_variable(
                name="rel_emb_{}".format(timestamp),
                shape=[self.batch_size * 2, self.internal_k],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32,
            )

            self.rel_emb = make_variable(
                name="rel_emb_{}".format(timestamp),
                shape=[len(self.rel_to_idx), self.internal_k],
                initializer=self.initializer.get_relation_initializer(),
                dtype=tf.float32,
            )

            # self.ent_emb = tf.compat.v1.get_variable('ent_emb_{}'.format(timestamp),
            #                                          shape=[self.batch_size * 2, self.internal_k],
            #                                          initializer=tf.zeros_initializer(),
            #                                          dtype=tf.float32)
            #
            # self.rel_emb = tf.compat.v1.get_variable('rel_emb_{}'.format(timestamp),
            #                                          shape=[len(self.rel_to_idx), self.internal_k],
            #                                          initializer=self.initializer.get_relation_initializer(
            #                                              len(self.rel_to_idx), self.internal_k),
            #                                          dtype=tf.float32)

    def _get_model_loss(self, dataset_iterator):
        """Get the current loss including loss wrt to regularization.
        If the model employs a combination of distinct losses, this function must be overridden (eg: VAE).

        :param dataset_iterator: Dataset iterator.
        :type dataset_iterator: tf.data.Iterator
        :return: The loss value that must be minimized.
        :rtype: tf.Tensor
        """
        # self.epoch = tf.placeholder(tf.float32)
        self.epoch = 0

        # self.batch_number = tf.placeholder(tf.int32)
        self.batch_number = 0
        if self.use_focusE:
            (
                x_pos_tf,
                self.unique_entities,
                ent_emb_batch,
                weights,
            ) = dataset_iterator.get_next()

        else:
            # get the train triples of the batch, unique entities and the corresponding embeddings
            # the latter 2 variables are passed only for large graphs.
            x_pos_tf, self.unique_entities, ent_emb_batch = dataset_iterator.get_next()

        # list of dependent ops that need to be evaluated before computing the loss
        dependencies = []

        # if the graph is large
        if self.dealing_with_large_graphs:
            # Create a dependency to load the embeddings of the batch entities dynamically
            init_ent_emb_batch = self.ent_emb.assign(ent_emb_batch, use_locking=True)
            dependencies.append(init_ent_emb_batch)

            # create a lookup dependency(to remap the entity indices to the corresponding indices of variables in memory
            self.sparse_mappings = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int32,
                value_dtype=tf.int32,
                default_value=-1,
                empty_key=-2,
                deleted_key=-1,
            )

            insert_lookup_op = self.sparse_mappings.insert(
                self.unique_entities,
                tf.reshape(
                    tf.range(tf.shape(self.unique_entities)[0], dtype=tf.int32), (-1, 1)
                ),
            )

            dependencies.append(insert_lookup_op)

        # run the dependencies
        with tf.control_dependencies(dependencies):
            entities_size = 0
            entities_list = None

            x_pos = x_pos_tf

            e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)

            scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

            non_linearity = self.embedding_model_params.get("non_linearity", "linear")
            if non_linearity == "linear":
                scores_pos = scores_pos
            elif non_linearity == "tanh":
                scores_pos = tf.tanh(scores_pos)
            elif non_linearity == "sigmoid":
                scores_pos = tf.sigmoid(scores_pos)
            elif non_linearity == "softplus":
                scores_pos = custom_softplus(scores_pos)
            else:
                raise ValueError("Invalid non-linearity")

            if self.use_focusE:

                epoch_before_stopping_weight = self.embedding_model_params.get(
                    "stop_epoch", 251
                )
                assert epoch_before_stopping_weight >= 0, "Invalid value for stop_epoch"

                if epoch_before_stopping_weight == 0:
                    # use fixed structural weight
                    structure_weight = self.embedding_model_params.get(
                        "structural_wt", 0.001
                    )
                    assert (
                        structure_weight <= 1 and structure_weight >= 0
                    ), "Invalid structure_weight passed to model params!"

                else:
                    # decay of numeric values
                    # start with all triples having same numeric values and linearly decay till original value
                    structure_weight = tf.maximum(
                        1 - self.epoch / epoch_before_stopping_weight, 0.001
                    )

                weights = tf.reduce_mean(weights, 1)
                weights_pos = structure_weight + (1 - structure_weight) * (1 - weights)
                weights_neg = structure_weight + (1 - structure_weight) * (
                    tf.reshape(
                        tf.tile(weights, [self.eta]), [tf.shape(weights)[0] * self.eta]
                    )
                )

                scores_pos = scores_pos * weights_pos

            if self.loss.get_state("require_same_size_pos_neg"):
                logger.debug("Requires the same size of postive and negative")
                scores_pos = tf.reshape(
                    tf.tile(scores_pos, [self.eta]),
                    [tf.shape(scores_pos)[0] * self.eta],
                )

            # look up embeddings from input training triples
            negative_corruption_entities = self.embedding_model_params.get(
                "negative_corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
            )

            if negative_corruption_entities == "all":
                """
                if number of entities are large then in this case('all'),
                the corruptions would be generated from batch entities and and additional random entities that
                are selected from all entities (since a total of batch_size*2 entity embeddings are loaded in memory)
                """
                logger.debug(
                    "Using all entities for generation of corruptions during training"
                )
                if self.dealing_with_large_graphs:
                    entities_list = tf.squeeze(self.unique_entities)
                else:
                    entities_size = tf.shape(self.ent_emb)[0]
            elif negative_corruption_entities == "batch":
                # default is batch (entities_size=0 and entities_list=None)
                logger.debug(
                    "Using batch entities for generation of corruptions during training"
                )
            elif isinstance(negative_corruption_entities, list):
                logger.debug(
                    "Using the supplied entities for generation of corruptions during training"
                )
                entities_list = tf.squeeze(
                    tf.constant(
                        np.asarray(
                            [
                                idx
                                for uri, idx in self.ent_to_idx.items()
                                if uri in negative_corruption_entities
                            ]
                        ),
                        dtype=tf.int32,
                    )
                )
            elif isinstance(negative_corruption_entities, int):
                logger.debug(
                    "Using first {} entities for generation of corruptions during \
                                                 training".format(
                        negative_corruption_entities
                    )
                )
                entities_size = negative_corruption_entities

            loss = 0
            corruption_sides = self.embedding_model_params.get(
                "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_TRAIN
            )
            if not isinstance(corruption_sides, list):
                corruption_sides = [corruption_sides]

            for side in corruption_sides:
                # Generate the corruptions
                x_neg_tf = generate_corruptions_for_fit(
                    x_pos_tf,
                    entities_list=entities_list,
                    eta=self.eta,
                    corrupt_side=side,
                    entities_size=entities_size,
                    rnd=self.seed,
                )

                # compute corruption scores
                e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)
                scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)

                if non_linearity == "linear":
                    scores_neg = scores_neg
                elif non_linearity == "tanh":
                    scores_neg = tf.tanh(scores_neg)
                elif non_linearity == "sigmoid":
                    scores_neg = tf.sigmoid(scores_neg)
                elif non_linearity == "softplus":
                    scores_neg = custom_softplus(scores_neg)
                else:
                    raise ValueError("Invalid non-linearity")

                if self.use_focusE:
                    scores_neg = scores_neg * weights_neg

                # Apply the loss function
                loss += self.loss.apply(scores_pos, scores_neg)

            if self.regularizer is not None:
                # Apply the regularizer
                loss += self.regularizer.apply([self.ent_emb, self.rel_emb])

            return loss

    def _initialize_early_stopping(self):
        """Initializes and creates evaluation graph for early stopping."""
        try:
            self.x_valid = self.early_stopping_params["x_valid"]

            if isinstance(self.x_valid, np.ndarray):
                if self.x_valid.ndim <= 1 or (np.shape(self.x_valid)[1]) != 3:
                    msg = "Invalid size for input x_valid. Expected (n,3):  got {}".format(
                        np.shape(self.x_valid)
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                # store the validation data in the data handler
                self.x_valid = to_idx(
                    self.x_valid, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx
                )
                self.train_dataset_handle.set_data(
                    self.x_valid, "valid", mapped_status=True
                )
                self.eval_dataset_handle = self.train_dataset_handle

            elif isinstance(self.x_valid, EmgraphBaseDatasetAdaptor):
                # this assumes that the validation data has already been set in the adapter
                self.eval_dataset_handle = self.x_valid
            else:
                msg = "Invalid type for input X. Expected ndarray/EmgraphDataset object, \
                       got {}".format(
                    type(self.x_valid)
                )
                logger.error(msg)
                raise ValueError(msg)
        except KeyError:
            msg = "x_valid must be passed for early fitting."
            logger.error(msg)
            raise KeyError(msg)

        self.early_stopping_criteria = self.early_stopping_params.get(
            "criteria", constants.DEFAULT_CRITERIA_EARLY_STOPPING
        )
        if self.early_stopping_criteria not in ["hits10", "hits1", "hits3", "mrr"]:
            msg = "Unsupported early stopping criteria."
            logger.error(msg)
            raise ValueError(msg)

        self.eval_config["corruption_entities"] = self.early_stopping_params.get(
            "corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
        )

        if isinstance(self.eval_config["corruption_entities"], list):
            # convert from list of raw triples to entity indices
            logger.debug(
                "Using the supplied entities for generation of corruptions for early stopping"
            )
            self.eval_config["corruption_entities"] = np.asarray(
                [
                    idx
                    for uri, idx in self.ent_to_idx.items()
                    if uri in self.eval_config["corruption_entities"]
                ]
            )
        elif self.eval_config["corruption_entities"] == "all":
            logger.debug(
                "Using all entities for generation of corruptions for early stopping"
            )
        elif self.eval_config["corruption_entities"] == "batch":
            logger.debug(
                "Using batch entities for generation of corruptions for early stopping"
            )

        self.eval_config["corrupt_side"] = self.early_stopping_params.get(
            "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_EVAL
        )

        self.early_stopping_best_value = None
        self.early_stopping_stop_counter = 0
        self.early_stopping_epoch = None

        try:
            # If the filter has already been set in the dataset adapter then just pass x_filter = True
            x_filter = self.early_stopping_params["x_filter"]
            if isinstance(x_filter, np.ndarray):
                if x_filter.ndim <= 1 or (np.shape(x_filter)[1]) != 3:
                    msg = "Invalid size for input x_valid. Expected (n,3):  got {}".format(
                        np.shape(x_filter)
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                # set the filter triples in the data handler
                x_filter = to_idx(
                    x_filter, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx
                )
                self.eval_dataset_handle.set_filter(x_filter, mapped_status=True)
            # set the flag to perform filtering
            self.set_filter_for_eval()
        except KeyError:
            logger.debug("x_filter not found in early_stopping_params.")
            pass

        # initialize evaluation graph in validation mode i.e. to use validation set
        self._initialize_eval_graph("valid")

    def _perform_early_stopping_test(self, epoch):
        """Performs regular validation checks and stops early if the criterion is met.

        :param epoch: current training epoch.
        :type epoch: int
        :return: Flag to indicate if the early stopping criteria is achieved.
        :rtype: bool
        """

        if (
            epoch
            >= self.early_stopping_params.get(
                "burn_in", constants.DEFAULT_BURN_IN_EARLY_STOPPING
            )
            and epoch
            % self.early_stopping_params.get(
                "check_interval", constants.DEFAULT_CHECK_INTERVAL_EARLY_STOPPING
            )
            == 0
        ):
            # compute and store test_loss
            ranks = []

            # Get each triple and compute the rank for that triple
            for x_test_triple in range(self.eval_dataset_handle.get_size("valid")):
                rank_triple = self.rank
                if (
                    self.eval_config.get(
                        "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_EVAL
                    )
                    == "s,o"
                ):
                    ranks.append(list(rank_triple))
                else:
                    ranks.append(rank_triple)

            if self.early_stopping_criteria == "hits10":
                current_test_value = hits_at_n_score(ranks, 10)
            elif self.early_stopping_criteria == "hits3":
                current_test_value = hits_at_n_score(ranks, 3)
            elif self.early_stopping_criteria == "hits1":
                current_test_value = hits_at_n_score(ranks, 1)
            elif self.early_stopping_criteria == "mrr":
                current_test_value = mrr_score(ranks)

            if self.tensorboard_logs_path is not None:
                tag = "Early stopping {} current value".format(
                    self.early_stopping_criteria
                )
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=current_test_value)]
                )
                self.writer.add_summary(summary, epoch)

            if self.early_stopping_best_value is None:  # First validation iteration
                self.early_stopping_best_value = current_test_value
                self.early_stopping_first_value = current_test_value
            elif self.early_stopping_best_value >= current_test_value:
                self.early_stopping_stop_counter += 1
                if self.early_stopping_stop_counter == self.early_stopping_params.get(
                    "stop_interval", constants.DEFAULT_STOP_INTERVAL_EARLY_STOPPING
                ):

                    # If the best value for the criteria has not changed from
                    #  initial value then
                    # save the model before early stopping
                    if (
                        self.early_stopping_best_value
                        == self.early_stopping_first_value
                    ):
                        self._save_trained_params()

                    if self.verbose:
                        msg = "Early stopping at epoch:{}".format(epoch)
                        logger.info(msg)
                        msg = "Best {}: {:10f}".format(
                            self.early_stopping_criteria, self.early_stopping_best_value
                        )
                        logger.info(msg)

                    self.early_stopping_epoch = epoch

                    return True
            else:
                self.early_stopping_best_value = current_test_value
                self.early_stopping_stop_counter = 0
                self._save_trained_params()

            if self.verbose:
                msg = "Current best:{}".format(self.early_stopping_best_value)
                logger.debug(msg)
                msg = "Current:{}".format(current_test_value)
                logger.debug(msg)

        return False

    def _after_training(self):
        """Performs clean up tasks after training."""
        # Reset this variable as it is reused during evaluation phase
        if self.is_filtered and self.eval_dataset_handle is not None:
            # cleanup the evaluation data (deletion of tables
            self.eval_dataset_handle.cleanup()
            self.eval_dataset_handle = None

        if self.train_dataset_handle is not None:
            self.train_dataset_handle.cleanup()
            self.train_dataset_handle = None

        self.is_filtered = False
        self.eval_config = {}

        # close the tf session
        # if self.sess_train is not None:
        #     self.sess_train.close()

        # set is_fitted to true to indicate that the model fitting is completed
        self.is_fitted = True

    def _training_data_generator(self):
        """Generates the training data.

        If we are working with huge graphs, this function returns the idx of the entities present in the batch (along
        with filler entities picked randomly from the rest (not in batch) to load batch size*2 entities on the GPU)
        and their embeddings, in addition to the training triples (of the batch).
        """

        all_ent = np.int32(np.arange(len(self.ent_to_idx)))
        unique_entities = all_ent.reshape(-1, 1)

        # generate empty embeddings for smaller graphs - as all the entity embeddings will be loaded on GPU
        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)

        # create iterator to iterate over the train batches
        batch_iterator = iter(
            self.train_dataset_handle.get_next_batch(self.batches_count, "train")
        )
        for i in range(self.batches_count):
            out = next(batch_iterator)

            out_triples = out[0]
            if self.use_focusE:
                out_weights = out[1]

            # If large graph, load batch_size*2 entities on GPU memory
            if self.dealing_with_large_graphs:
                # find the unique entities - these HAVE to be loaded
                unique_entities = np.int32(
                    np.unique(
                        np.concatenate([out_triples[:, 0], out_triples[:, 2]], axis=0)
                    )
                )
                # Load the remaining entities by randomly selecting from the rest of the entities
                self.leftover_entities = self.rnd.permutation(
                    np.setdiff1d(all_ent, unique_entities)
                )
                needed = self.batch_size * 2 - unique_entities.shape[0]
                """
                #this is for debugging
                large_number = np.zeros((self.batch_size-unique_entities.shape[0],
                                             self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan

                entity_embeddings = np.concatenate((self.ent_emb_cpu[unique_entities,:],
                                                    large_number), axis=0)
                """
                unique_entities = np.int32(
                    np.concatenate(
                        [unique_entities, self.leftover_entities[:needed]], axis=0
                    )
                )
                entity_embeddings = self.ent_emb_cpu[unique_entities, :]

                unique_entities = unique_entities.reshape(-1, 1)

            if self.use_focusE:
                for col_idx in range(out_weights.shape[1]):
                    # random weights are used where weights are unknown
                    nan_indices = np.isnan(out_weights[:, col_idx])
                    out_weights[nan_indices, col_idx] = np.random.uniform(
                        size=(np.sum(nan_indices))
                    )

                out_weights = np.mean(out_weights, 1)
                out_weights = out_weights[:, np.newaxis]
                yield out_triples, unique_entities, entity_embeddings, out_weights
            else:
                yield np.squeeze(out_triples), unique_entities, entity_embeddings

    def fit(
        self,
        X,
        early_stopping=False,
        early_stopping_params={},
        focusE_numeric_edge_values=None,
        tensorboard_logs_path=None,
    ):
        """Train an EmbeddingModel (with optional early stopping).

        The model is trained on a training set X using the training protocol described in :cite:`trouillon2016complex`.

        :param X: Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        :type X: ndarray (shape [n, 3]) or object of EmgraphBaseDatasetAdaptor
        :param early_stopping: Flag to enable early stopping (default:``False``)
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
            Semantically, the numeric value can signify importance, uncertainity, significance, confidence, etc.
            If the numeric value is unknown pass a NaN weight. The model will uniformly randomly assign a numeric value.
            One can also think about assigning numeric values by looking at the distribution of it per predicate.
        :type focusE_numeric_edge_values: nd array (n, 1)
        :param tensorboard_logs_path: Path to store tensorboard logs, e.g. average training loss tracking per epoch (default: ``None`` indicating
            no logs will be collected). When provided it will create a folder under provided path and save tensorboard
            files there. To then view the loss in the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
        :type tensorboard_logs_path: str or None
        :return:
        :rtype:
        """

        self.train_dataset_handle = None
        self.tensorboard_logs_path = tensorboard_logs_path
        # In a jupyter notebook, the try-except block is mostly used to clean up after an exception or a manual stop.
        try:
            if isinstance(X, np.ndarray):
                if focusE_numeric_edge_values is not None:
                    logger.debug("Using FocusE")
                    self.use_focusE = True
                    assert focusE_numeric_edge_values.shape[0] == X.shape[0], (
                        "Each triple must have a numeric value (the size of the training set does not match the size"
                        "of the focusE_numeric_edge_values argument."
                    )

                    if focusE_numeric_edge_values.ndim == 1:
                        focusE_numeric_edge_values = focusE_numeric_edge_values.reshape(
                            -1, 1
                        )

                    logger.debug("normalizing numeric values")
                    unique_relations = np.unique(X[:, 1])
                    for reln in unique_relations:
                        for col_idx in range(focusE_numeric_edge_values.shape[1]):
                            # here nans signify unknown numeric values
                            if (
                                np.sum(
                                    np.isnan(
                                        focusE_numeric_edge_values[
                                            X[:, 1] == reln, col_idx
                                        ]
                                    )
                                )
                                != focusE_numeric_edge_values[
                                    X[:, 1] == reln, col_idx
                                ].shape[0]
                            ):
                                min_val = np.nanmin(
                                    focusE_numeric_edge_values[X[:, 1] == reln, col_idx]
                                )
                                max_val = np.nanmax(
                                    focusE_numeric_edge_values[X[:, 1] == reln, col_idx]
                                )
                                if min_val == max_val:
                                    focusE_numeric_edge_values[
                                        X[:, 1] == reln, col_idx
                                    ] = 1.0
                                    continue

                                if (
                                    self.embedding_model_params.get(
                                        "normalize_numeric_values", True
                                    )
                                    or min_val < 0
                                    or max_val > 1
                                ):
                                    focusE_numeric_edge_values[
                                        X[:, 1] == reln, col_idx
                                    ] = (
                                        focusE_numeric_edge_values[
                                            X[:, 1] == reln, col_idx
                                        ]
                                        - min_val
                                    ) / (
                                        max_val - min_val
                                    )
                            else:
                                pass  # all the weights are nans

                # Adapt the numpy data in the internal format - to generalize
                self.train_dataset_handle = NumpyDatasetAdapter()
                self.train_dataset_handle.set_data(
                    X, "train", focusE_numeric_edge_values=focusE_numeric_edge_values
                )
            elif isinstance(X, EmgraphBaseDatasetAdaptor):
                self.train_dataset_handle = X
            else:
                msg = "Invalid type for input X. Expected ndarray/EmgraphDataset object, got {}".format(
                    type(X)
                )
                logger.error(msg)
                raise ValueError(msg)

            # create internal IDs mappings
            (
                self.rel_to_idx,
                self.ent_to_idx,
            ) = self.train_dataset_handle.generate_mappings()
            prefetch_batches = 1

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                self.dealing_with_large_graphs = True

                logger.warning(
                    "Your graph has a large number of distinct entities. "
                    "Found {} distinct entities".format(len(self.ent_to_idx))
                )

                logger.warning("Changing the variable initialization strategy.")
                logger.warning(
                    "Changing the strategy to use lazy loading of variables..."
                )

                if early_stopping:
                    raise Exception("Early stopping not supported for large graphs")

                if not isinstance(self.optimizer, SGD):
                    raise Exception(
                        "This mode works well only with SGD optimizer with decay."
                        "Kindly change the optimizer and restart the experiment. For details refer the "
                        "following link: \n"
                        # todo: remove this
                        "https://docs.emgraph.org/en/latest/dev_notes.html#dealing-with-large-graphs"
                    )

            if self.dealing_with_large_graphs:
                prefetch_batches = 0
                # CPU matrix of embeddings
                self.ent_emb_cpu = self.initializer.get_entity_initializer(
                    len(self.ent_to_idx), self.internal_k, "np"
                )

            self.train_dataset_handle.map_data()

            # This is useful when we re-fit the same model (e.g. retraining in model selection)
            if self.is_fitted:
                # tf.reset_default_graph()
                self.rnd = check_random_state(self.seed)
                # tf.random.set_random_seed(self.seed)
                tf.random.set_seed(self.seed)

            # self.sess_train = tf.Session(config=self.tf_config)
            # print('self.sess train: ', self.sess_train)
            if self.tensorboard_logs_path is not None:
                # self.writer = tf.summary.FileWriter(self.tensorboard_logs_path, self.sess_train.graph)
                self.writer = tf.summary.create_file_writer(self.tensorboard_logs_path)
            batch_size = int(
                np.ceil(
                    self.train_dataset_handle.get_size("train") / self.batches_count
                )
            )
            # dataset = tf.data.Dataset.from_tensor_slices(X).repeat().batch(batch_size).prefetch(2)

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                logger.warning(
                    "Only {} embeddings would be loaded in memory per batch...".format(
                        batch_size * 2
                    )
                )

            self.batch_size = batch_size
            self._initialize_parameters()

            if self.use_focusE:
                output_types = (tf.int32, tf.int32, tf.float32, tf.float32)
                output_shapes = (
                    (None, 3),
                    (None, 1),
                    (None, self.internal_k),
                    (None, 1),
                )
            else:
                output_types = (tf.int32, tf.int32, tf.float32)
                output_shapes = ((None, 3), (None, 1), (None, self.internal_k))

            # todo ============================= Starts from here =============================================================
            # todo ============================================================================================================

            dataset = tf.data.Dataset.from_generator(
                self._training_data_generator,
                output_types=output_types,
                output_shapes=output_shapes,
            )
            dataset = dataset.repeat().prefetch(prefetch_batches)
            # print("dataset: ", dataset)

            dataset_iterator = dataset.__iter__()
            # dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

            # init tf graph/dataflow for training
            # init variables (model parameters to be learned - i.e. the embeddings)

            if self.loss.get_state("require_same_size_pos_neg"):
                batch_size = batch_size * self.eta

            loss = functools.partial(self._get_model_loss, dataset_iterator)
            # loss = self._get_model_loss(dataset_iterator)
            # print(loss)
            # todo: change this to get other models' hyperparameters instead of TrnasE
            print("model: ", MODEL_REGISTRY["TransE"].get_hyperparameter_dict(self))
            print("optimizer: ", self.optimizer)
            print("OPTIMIZER_REGISTRY: ", OPTIMIZER_REGISTRY)

            # train = self.optimizer.minimize(loss, [self.ent_emb, self.rel_emb])

            self.early_stopping_params = early_stopping_params

            # early stopping
            if early_stopping:
                self._initialize_early_stopping()

            # tf2x_dev changes:
            # self.sess_train.run(tf.tables_initializer())
            # self.sess_train.run(tf.global_variables_initializer())
            # try:
            #     self.sess_train.run(self.set_training_true)
            # except AttributeError:
            #     pass

            # todo: this is tf1 compatible -> should be removed
            if self.embedding_model_params.get(
                "normalize_ent_emb", constants.DEFAULT_NORMALIZE_EMBEDDINGS
            ):
                # Entity embeddings normalization
                normalize_rel_emb_op = self.rel_emb.assign(
                    tf.clip_by_norm(self.rel_emb, clip_norm=1, axes=1)
                )
                normalize_ent_emb_op = self.ent_emb.assign(
                    tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1)
                )
                # self.sess_train.run(normalize_rel_emb_op)
                # self.sess_train.run(normalize_ent_emb_op)

            epoch_iterator_with_progress = tqdm(
                range(1, self.epochs + 1), disable=(not self.verbose), unit="epoch"
            )
            # print("epoch_iterator_with_progress: ", epoch_iterator_with_progress)
            for epoch in epoch_iterator_with_progress:
                losses = []
                for batch in range(1, self.batches_count + 1):
                    # feed_dict = {self.epoch: epoch, self.batch_number: batch - 1}
                    # print("feed_dict: ", feed_dict)
                    # print("feed_dict: ", self.epoch, epoch)
                    # print("feed_dict: ", self.batch_number, batch - 1)
                    # self.optimizer.update_feed_dict(feed_dict, batch, epoch)
                    # print("self.optimizer.update_feed_dict: ", self.optimizer.update_feed_dict)
                    if self.dealing_with_large_graphs:
                        # loss_batch, unique_entities, _ = self.sess_train.run([loss, self.unique_entities, train],
                        #                                                      feed_dict=feed_dict)
                        loss_batch = loss
                        unique_entities = self.unique_entities
                        train = self.optimizer.minimize(
                            loss_batch, [self.ent_emb, self.rel_emb]
                        )
                        self.ent_emb_cpu[np.squeeze(unique_entities), :] = self.ent_emb[
                            : unique_entities.shape[0], :
                        ]
                        # self.sess_train.run(self.ent_emb)[:unique_entities.shape[0], :]
                    else:
                        # print("loss: ", loss)
                        # print("train: ", train)

                        loss_batch = partial(self._get_model_loss, dataset_iterator)
                        train = self.optimizer.minimize(
                            loss_batch, [self.ent_emb, self.rel_emb]
                        )
                        # print("loss_batch: ", loss_batch.numpy())
                        # loss_batch, _ = self.sess_train.run([loss, train], feed_dict=feed_dict)
                        loss_batch = self._get_model_loss(dataset_iterator).numpy()
                    if np.isnan(loss_batch) or np.isinf(loss_batch):
                        msg = "Loss is {}. Please change the hyperparameters.".format(
                            loss_batch
                        )
                        logger.error(msg)
                        raise ValueError(msg)

                    losses.append(loss_batch)

                    # todo ============================= Ends here ====================================================================
                    # todo ============================================================================================================

                    if self.embedding_model_params.get(
                        "normalize_ent_emb", constants.DEFAULT_NORMALIZE_EMBEDDINGS
                    ):
                        normalize_ent_emb_op = self.ent_emb.assign(
                            tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1)
                        )
                        # self.sess_train.run(normalize_ent_emb_op)
                if self.tensorboard_logs_path is not None:
                    avg_loss = sum(losses) / (batch_size * self.batches_count)
                    summary = tf.Summary(
                        value=[
                            tf.Summary.Value(tag="Average Loss", simple_value=avg_loss)
                        ]
                    )
                    self.writer.add_summary(summary, epoch)
                if self.verbose:
                    focusE = ""
                    if self.use_focusE:
                        focusE = "-FocusE"
                    msg = "Average {}{} Loss: {:10f}".format(
                        self.name,
                        focusE,
                        sum(losses) / (batch_size * self.batches_count),
                    )
                    if early_stopping and self.early_stopping_best_value is not None:
                        msg += " — Best validation ({}): {:5f}".format(
                            self.early_stopping_criteria, self.early_stopping_best_value
                        )

                    logger.debug(msg)
                    epoch_iterator_with_progress.set_description(msg)

                if early_stopping:

                    # try:
                    #     self.sess_train.run(self.set_training_false)
                    # except AttributeError:
                    #     pass

                    if self._perform_early_stopping_test(epoch):
                        if self.tensorboard_logs_path is not None:
                            self.writer.flush()
                            self.writer.close()
                        self._after_training()
                        return

                    try:
                        self.set_training_true
                    except AttributeError:
                        pass
            if self.tensorboard_logs_path is not None:
                self.writer.flush()
                self.writer.close()

            self._save_trained_params()
            self._after_training()
        except BaseException as e:
            self._after_training()
            raise e

    def set_filter_for_eval(self):
        """Configures to use filter"""
        self.is_filtered = True

    def configure_evaluation_protocol(self, config=None):
        """Set the configuration for evaluation

        :param config: Dictionary of parameters for evaluation configuration. Can contain following keys:

            - **corruption_entities**: List of entities to be used for corruptions.
              If ``all``, it uses all entities (default: ``all``)
            - **corrupt_side**: Specifies which side to corrupt. ``s``, ``o``, ``s+o``, ``s,o`` (default)
              In 's,o' mode subject and object corruptions are generated at once but ranked separately
              for speed up (default: False).
        :type config: dict
        :return:
        :rtype:
        """

        if config is None:
            config = {
                "corruption_entities": constants.DEFAULT_CORRUPTION_ENTITIES,
                "corrupt_side": constants.DEFAULT_CORRUPT_SIDE_EVAL,
            }
        self.eval_config = config

    def _test_generator(self, mode):
        """Generates the test/validation data. If filter_triples are passed, then it returns the False Negatives
           that could be present in the generated corruptions.

           If we are working with big graphs, this function also returns the idx of the node. Entities in the batch
           as well as their embeddings.

        :param mode: Dataset type
        :type mode: str
        :return:
        :rtype:
        """

        test_generator = partial(
            self.eval_dataset_handle.get_next_batch,
            dataset_type=mode,
            use_filter=self.is_filtered,
        )

        batch_iterator = iter(test_generator())
        indices_obj = np.empty(shape=(0, 1), dtype=np.int32)
        indices_sub = np.empty(shape=(0, 1), dtype=np.int32)
        unique_ent = np.empty(shape=(0, 1), dtype=np.int32)
        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)
        for i in range(self.eval_dataset_handle.get_size(mode)):
            if self.is_filtered:
                out, indices_obj, indices_sub = next(batch_iterator)
            else:
                out = next(batch_iterator)
                # since focuse layer is not used in evaluation mode
                out = out[0]

            if self.dealing_with_large_graphs:
                # since we are dealing with only one triple (2 entities)
                unique_ent = np.unique(np.array([out[0, 0], out[0, 2]]))
                needed = self.corr_batch_size - unique_ent.shape[0]
                large_number = (
                    np.zeros((needed, self.ent_emb_cpu.shape[1]), dtype=np.float32)
                    + np.nan
                )
                entity_embeddings = np.concatenate(
                    (self.ent_emb_cpu[unique_ent, :], large_number), axis=0
                )
                unique_ent = unique_ent.reshape(-1, 1)

            yield out, indices_obj, indices_sub, entity_embeddings, unique_ent

    def _generate_corruptions_for_large_graphs(self):
        """
        Corruption generator for large graph mode only.

        It generates corruptions in batches as well as the related entity embeddings.
        """

        corruption_entities = self.eval_config.get(
            "corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
        )

        if corruption_entities == "all":
            all_entities_np = np.arange(len(self.ent_to_idx))
            corruption_entities = all_entities_np
        elif isinstance(corruption_entities, np.ndarray):
            corruption_entities = corruption_entities
        else:
            msg = "Invalid type for corruption entities."
            logger.error(msg)
            raise ValueError(msg)

        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)

        for i in range(self.corr_batches_count):
            all_ent = corruption_entities[
                i * self.corr_batch_size : (i + 1) * self.corr_batch_size
            ]
            needed = self.corr_batch_size - all_ent.shape[0]
            large_number = (
                np.zeros((needed, self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan
            )
            entity_embeddings = np.concatenate(
                (self.ent_emb_cpu[all_ent, :], large_number), axis=0
            )

            all_ent = all_ent.reshape(-1, 1)
            yield all_ent, entity_embeddings

    def _initialize_eval_graph(self, mode="test"):
        """Initialize the evaluation graph.

        :param mode: Indicates which data generator to use.
        :type mode: str
        :return:
        :rtype:
        """

        # If the graph is large, use a data generator that returns a test triple as well as the subjects and objects
        # indices for filtering. They are the embeddings of the entities that must be loaded on the GPU before scoring
        # and the indices of those embeddings.
        dataset = tf.data.Dataset.from_generator(
            partial(self._test_generator, mode=mode),
            output_types=(tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
            output_shapes=(
                (1, 3),
                (None, 1),
                (None, 1),
                (None, self.internal_k),
                (None, 1),
            ),
        )
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)
        dataset_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
        (
            self.X_test_tf,
            indices_obj,
            indices_sub,
            entity_embeddings,
            unique_ent,
        ) = dataset_iter.get_next()

        corrupt_side = self.eval_config.get(
            "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_EVAL
        )

        # Rather than generating corruptions in batches do it at once on the GPU for small or medium sized graphs
        all_entities_np = np.arange(len(self.ent_to_idx))

        corruption_entities = self.eval_config.get(
            "corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
        )

        if corruption_entities == "all":
            corruption_entities = all_entities_np
        elif isinstance(corruption_entities, np.ndarray):
            corruption_entities = corruption_entities
        else:
            msg = "Invalid type for corruption entities."
            logger.error(msg)
            raise ValueError(msg)

        # Dependencies that need to be run before scoring
        test_dependency = []
        # For large graphs
        if self.dealing_with_large_graphs:
            # Add a dependency to load the embeddings on the GPU
            init_ent_emb_batch = self.ent_emb.assign(
                entity_embeddings, use_locking=True
            )
            test_dependency.append(init_ent_emb_batch)

            # Add a dependency to create lookup tables(for remapping the entity indices to the order of variables on GPU
            self.sparse_mappings = tf.compat.v1.raw_ops.MutableDenseHashTable(
                key_dtype=tf.int32,
                value_dtype=tf.int32,
                default_value=-1,
                empty_key=-2,
                deleted_key=-1,
            )
            insert_lookup_op = self.sparse_mappings.insert(
                unique_ent,
                tf.reshape(tf.range(tf.shape(unique_ent)[0], dtype=tf.int32), (-1, 1)),
            )
            test_dependency.append(insert_lookup_op)
            if isinstance(corruption_entities, np.ndarray):
                # This is used to map the corruption entities' scores to the array that stores the scores. Because
                # the number of entities is low when entities subset is utilized, the size of the array that stores
                # the scores is len(entities subset). As a result, when storing, the corrupted object id must be
                # mapped to an array index.
                rankings_mappings = tf.compat.v1.raw_ops.MutableDenseHashTable(
                    key_dtype=tf.int32,
                    value_dtype=tf.int32,
                    default_value=-1,
                    empty_key=-2,
                    deleted_key=-1,
                )

                ranking_lookup_op = rankings_mappings.insert(
                    corruption_entities.reshape(-1, 1),
                    tf.reshape(
                        tf.range(len(corruption_entities), dtype=tf.int32), (-1, 1)
                    ),
                )
                test_dependency.append(ranking_lookup_op)

            # Execute the dependency
            with tf.control_dependencies(test_dependency):
                # Compute scores for positive - single triple
                e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
                self.score_positive = tf.squeeze(self._fn(e_s, e_p, e_o))

                # Generate corruptions in batches
                self.corr_batches_count = int(
                    np.ceil(len(corruption_entities) / (self.corr_batch_size))
                )

                # Corruption generator -
                # returns corruptions and their corresponding embeddings that need to be loaded on the GPU
                corruption_generator = tf.data.Dataset.from_generator(
                    self._generate_corruptions_for_large_graphs,
                    output_types=(tf.int32, tf.float32),
                    output_shapes=((None, 1), (None, self.internal_k)),
                )

                corruption_generator = corruption_generator.repeat()
                corruption_generator = corruption_generator.prefetch(0)

                corruption_iter = tf.compat.v1.data.make_one_shot_iterator(
                    corruption_generator
                )

                # Create tensor arrays for storing the scores of subject and object evals
                # size of this array must be equal to size of entities used for corruption.
                scores_predict_s_corruptions = tf.TensorArray(
                    dtype=tf.float32, size=(len(corruption_entities))
                )
                scores_predict_o_corruptions = tf.TensorArray(
                    dtype=tf.float32, size=(len(corruption_entities))
                )

                def loop_cond(
                    i, scores_predict_s_corruptions_in, scores_predict_o_corruptions_in
                ):
                    return i < self.corr_batches_count

                def compute_score_corruptions(
                    i, scores_predict_s_corruptions_in, scores_predict_o_corruptions_in
                ):
                    corr_dependency = []
                    corr_batch, entity_embeddings_corrpt = corruption_iter.get_next()
                    # if self.dealing_with_large_graphs: #for debugging
                    # Add dependency to load the embeddings
                    init_ent_emb_corrpt = self.ent_emb.assign(
                        entity_embeddings_corrpt, use_locking=True
                    )
                    corr_dependency.append(init_ent_emb_corrpt)

                    # Add dependency to remap the indices to the corresponding indices on the GPU
                    insert_lookup_op2 = self.sparse_mappings.insert(
                        corr_batch,
                        tf.reshape(
                            tf.range(tf.shape(corr_batch)[0], dtype=tf.int32), (-1, 1)
                        ),
                    )
                    corr_dependency.append(insert_lookup_op2)
                    # end if

                    # Execute the dependency
                    with tf.control_dependencies(corr_dependency):
                        emb_corr = tf.squeeze(self._entity_lookup(corr_batch))
                        if isinstance(corruption_entities, np.ndarray):
                            remapping = rankings_mappings.lookup(corr_batch)
                        else:
                            remapping = corr_batch
                        if "s" in corrupt_side:
                            # compute and store the scores batch wise
                            scores_predict_s_c = self._fn(emb_corr, e_p, e_o)
                            scores_predict_s_corruptions_in = (
                                scores_predict_s_corruptions_in.scatter(
                                    tf.squeeze(remapping),
                                    tf.squeeze(scores_predict_s_c),
                                )
                            )

                        if "o" in corrupt_side:
                            scores_predict_o_c = self._fn(e_s, e_p, emb_corr)
                            scores_predict_o_corruptions_in = (
                                scores_predict_o_corruptions_in.scatter(
                                    tf.squeeze(remapping),
                                    tf.squeeze(scores_predict_o_c),
                                )
                            )

                    return (
                        i + 1,
                        scores_predict_s_corruptions_in,
                        scores_predict_o_corruptions_in,
                    )

                # compute the scores for all the corruptions
                (
                    counter,
                    scores_predict_s_corr_out,
                    scores_predict_o_corr_out,
                ) = tf.while_loop(
                    loop_cond,
                    compute_score_corruptions,
                    (0, scores_predict_s_corruptions, scores_predict_o_corruptions),
                    back_prop=False,
                    parallel_iterations=1,
                )

                if "s" in corrupt_side:
                    subj_corruption_scores = scores_predict_s_corr_out.stack()

                if "o" in corrupt_side:
                    obj_corruption_scores = scores_predict_o_corr_out.stack()

                non_linearity = self.embedding_model_params.get(
                    "non_linearity", "linear"
                )
                if non_linearity == "linear":
                    pass
                elif non_linearity == "tanh":
                    subj_corruption_scores = tf.tanh(subj_corruption_scores)
                    obj_corruption_scores = tf.tanh(obj_corruption_scores)
                    self.score_positive = tf.tanh(self.score_positive)
                elif non_linearity == "sigmoid":
                    subj_corruption_scores = tf.sigmoid(subj_corruption_scores)
                    obj_corruption_scores = tf.sigmoid(obj_corruption_scores)
                    self.score_positive = tf.sigmoid(self.score_positive)
                elif non_linearity == "softplus":
                    subj_corruption_scores = custom_softplus(subj_corruption_scores)
                    obj_corruption_scores = custom_softplus(obj_corruption_scores)
                    self.score_positive = custom_softplus(self.score_positive)
                else:
                    raise ValueError("Invalid non-linearity")

                if corrupt_side == "s+o" or corrupt_side == "s,o":
                    self.scores_predict = tf.concat(
                        [obj_corruption_scores, subj_corruption_scores], axis=0
                    )
                elif corrupt_side == "o":
                    self.scores_predict = obj_corruption_scores
                else:
                    self.scores_predict = subj_corruption_scores

        else:

            # Entities that must be used while generating corruptions
            self.corruption_entities_tf = tf.constant(
                corruption_entities, dtype=tf.int32
            )

            corrupt_side = self.eval_config.get(
                "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_EVAL
            )
            # Generate corruptions
            self.out_corr = generate_corruptions_for_eval(
                self.X_test_tf, self.corruption_entities_tf, corrupt_side
            )

            # Compute scores for negatives
            e_s, e_p, e_o = self._lookup_embeddings(self.out_corr)
            self.scores_predict = self._fn(e_s, e_p, e_o)

            # Compute scores for positive
            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
            self.score_positive = tf.squeeze(self._fn(e_s, e_p, e_o))

            non_linearity = self.embedding_model_params.get("non_linearity", "linear")
            if non_linearity == "linear":
                pass
            elif non_linearity == "tanh":
                self.score_positive = tf.tanh(self.score_positive)
                self.scores_predict = tf.tanh(self.scores_predict)
            elif non_linearity == "sigmoid":
                self.score_positive = tf.sigmoid(self.score_positive)
                self.scores_predict = tf.sigmoid(self.scores_predict)
            elif non_linearity == "softplus":
                self.score_positive = custom_softplus(self.score_positive)
                self.scores_predict = custom_softplus(self.scores_predict)
            else:
                raise ValueError("Invalid non-linearity")

            if corrupt_side == "s,o":
                obj_corruption_scores = tf.slice(
                    self.scores_predict, [0], [tf.shape(self.scores_predict)[0] // 2]
                )

                subj_corruption_scores = tf.slice(
                    self.scores_predict,
                    [tf.shape(self.scores_predict)[0] // 2],
                    [tf.shape(self.scores_predict)[0] // 2],
                )

        # this is to remove the positives from corruptions - while ranking with filter
        positives_among_obj_corruptions_ranked_higher = tf.constant(0, dtype=tf.int32)
        positives_among_sub_corruptions_ranked_higher = tf.constant(0, dtype=tf.int32)

        if self.is_filtered:
            # If a list of specified entities were used for corruption generation
            if isinstance(
                self.eval_config.get(
                    "corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
                ),
                np.ndarray,
            ):
                corruption_entities = self.eval_config.get(
                    "corruption_entities", constants.DEFAULT_CORRUPTION_ENTITIES
                ).astype(np.int32)
                if corruption_entities.ndim == 1:
                    corruption_entities = np.expand_dims(corruption_entities, 1)
                # If the specified key is not present then it would return the length of corruption_entities
                corruption_mapping = tf.contrib.lookup.MutableDenseHashTable(
                    key_dtype=tf.int32,
                    value_dtype=tf.int32,
                    default_value=len(corruption_entities),
                    empty_key=-2,
                    deleted_key=-1,
                )

                insert_lookup_op = corruption_mapping.insert(
                    corruption_entities,
                    tf.reshape(
                        tf.range(tf.shape(corruption_entities)[0], dtype=tf.int32),
                        (-1, 1),
                    ),
                )

                with tf.control_dependencies([insert_lookup_op]):
                    # remap the indices of objects to the smaller set of corruptions
                    indices_obj = corruption_mapping.lookup(indices_obj)
                    # mask out the invalid indices (i.e. the entities that were not in corruption list
                    indices_obj = tf.boolean_mask(
                        indices_obj, indices_obj < len(corruption_entities)
                    )
                    # remap the indices of subject to the smaller set of corruptions
                    indices_sub = corruption_mapping.lookup(indices_sub)
                    # mask out the invalid indices (i.e. the entities that were not in corruption list
                    indices_sub = tf.boolean_mask(
                        indices_sub, indices_sub < len(corruption_entities)
                    )

            # get the scores of positives present in corruptions
            if corrupt_side == "s,o":
                scores_pos_obj = tf.gather(obj_corruption_scores, indices_obj)
                scores_pos_sub = tf.gather(subj_corruption_scores, indices_sub)
            else:
                scores_pos_obj = tf.gather(self.scores_predict, indices_obj)
                if corrupt_side == "s+o":
                    scores_pos_sub = tf.gather(
                        self.scores_predict, indices_sub + len(corruption_entities)
                    )
                else:
                    scores_pos_sub = tf.gather(self.scores_predict, indices_sub)
            # compute the ranks of the positives present in the corruptions and
            # see how many are ranked higher than the test triple
            if "o" in corrupt_side:
                positives_among_obj_corruptions_ranked_higher = (
                    self.perform_comparision(scores_pos_obj, self.score_positive)
                )
            if "s" in corrupt_side:
                positives_among_sub_corruptions_ranked_higher = (
                    self.perform_comparision(scores_pos_sub, self.score_positive)
                )

        # compute the rank of the test triple and subtract the positives(from corruptions) that are ranked higher
        if corrupt_side == "s,o":
            self.rank = tf.stack(
                [
                    self.perform_comparision(
                        subj_corruption_scores, self.score_positive
                    )
                    + 1
                    - positives_among_sub_corruptions_ranked_higher,
                    self.perform_comparision(obj_corruption_scores, self.score_positive)
                    + 1
                    - positives_among_obj_corruptions_ranked_higher,
                ],
                0,
            )
        else:
            self.rank = (
                self.perform_comparision(self.scores_predict, self.score_positive)
                + 1
                - positives_among_sub_corruptions_ranked_higher
                - positives_among_obj_corruptions_ranked_higher
            )

    # todo: change the comparision to comparison
    def perform_comparision(self, score_corr, score_pos):
        """Compare the scores of corruptions and positives using the specified strategy.

        :param score_corr: Scores of corruptions
        :type score_corr: tf.Tensor
        :param score_pos: Score of positive triple
        :type score_pos: tf.Tensor
        :return: comparison output based on specified strategy
        :rtype: int
        """

        comparision_type = self.eval_config.get(
            "ranking_strategy", constants.DEFAULT_RANK_COMPARE_STRATEGY
        )

        assert comparision_type in [
            "worst",
            "best",
            "middle",
        ], "Invalid score comparision type!"

        score_corr = tf.cast(
            score_corr * constants.SCORE_COMPARISON_PRECISION, tf.int32
        )

        score_pos = tf.cast(score_pos * constants.SCORE_COMPARISON_PRECISION, tf.int32)

        # if pos score: 0.5, corr_score: 0.5, 0.5, 0.3, 0.6, 0.5, 0.5
        if comparision_type == "best":
            # returns: 1 i.e. only. 1 corruption is having score greater than positive (optimistic)
            return tf.reduce_sum(tf.cast(score_corr > score_pos, tf.int32))
        elif comparision_type == "middle":

            # returns: 3 i.e. 1 + (4/2) i.e. only 1  corruption is having score greater than positive
            # and 4 corruptions are having same (middle rank is 4/2 = 1), so 1+2=3
            return tf.reduce_sum(tf.cast(score_corr > score_pos, tf.int32)) + tf.cast(
                tf.math.ceil(
                    tf.reduce_sum(tf.cast(score_corr == score_pos, tf.int32)) / 2
                ),
                tf.int32,
            )
        else:
            # returns: 5 i.e. 5 corruptions are having score >= positive
            # as you can see this strategy returns the worst rank (pessimistic)
            return tf.reduce_sum(tf.cast(score_corr >= score_pos, tf.int32))

    def end_evaluation(self):
        """End the evaluation."""

        if self.is_filtered and self.eval_dataset_handle is not None:
            self.eval_dataset_handle.cleanup()
            self.eval_dataset_handle = None

        self.is_filtered = False

        self.eval_config = {}

    def get_ranks(self, dataset_handle):
        """Used by evaluate_predictions to get the ranks for evaluation.

        :param dataset_handle: This contains handles of the generators that would be used to get test triples and filters
        :type dataset_handle: Object of EmgraphBaseDatasetAdaptor
        :return: An array of ranks of test triples
        :rtype: ndarray, shape [n] or [n,2] depending on the value of corrupt_side
        """

        if not self.is_fitted:
            msg = "Model has not been fitted."
            logger.error(msg)
            raise RuntimeError(msg)

        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        # tf.reset_default_graph()
        self.rnd = check_random_state(self.seed)
        # tf.random.set_random_seed(self.seed)
        tf.random.set_seed(self.seed)
        # load the parameters
        self._load_model_from_trained_params()
        # build the eval graph
        self._initialize_eval_graph()

        # with tf.Session(config=self.tf_config) as sess:
        # sess.run(tf.tables_initializer())
        # sess.run(tf.global_variables_initializer())

        try:
            # sess.run(self.set_training_false)
            self.set_training_false
        except AttributeError:
            pass

        ranks = []

        for _ in tqdm(
            range(self.eval_dataset_handle.get_size("test")), disable=(not self.verbose)
        ):
            # rank = sess.run(self.rank)
            rank = self.rank
            if (
                self.eval_config.get(
                    "corrupt_side", constants.DEFAULT_CORRUPT_SIDE_EVAL
                )
                == "s,o"
            ):
                ranks.append(list(rank))
            else:
                ranks.append(rank)

        return ranks

    def predict(self, X, from_idx=False):
        """Predict the scores of triples using a trained embedding model.
        The function returns raw scores generated by the model.

        .. note:: To obtain probability estimates, calibrate the model with :func:`~EmbeddingModel.calibrate`,
        then call :func:`~EmbeddingModel.predict_proba`.

        :param X: The triples to score.
        :type X: ndarray, shape [n, 3]
        :param from_idx: If True, will skip conversion to internal IDs. (default: False).
        :type from_idx: bool
        :return: The predicted scores for input triples X.
        :rtype: ndarray, shape [n]
        """

        if not self.is_fitted:
            msg = "Model has not been fitted."
            logger.error(msg)
            raise RuntimeError(msg)

        # tf.reset_default_graph()
        self._load_model_from_trained_params()

        if type(X) is not np.ndarray:
            X = np.array(X)

        if not self.dealing_with_large_graphs:
            if not from_idx:
                X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
            x_tf = tf.Variable(X, dtype=tf.int32, trainable=False)

            e_s, e_p, e_o = self._lookup_embeddings(x_tf)
            scores = self._fn(e_s, e_p, e_o)

            non_linearity = self.embedding_model_params.get("non_linearity", "linear")
            if non_linearity == "linear":
                pass
            elif non_linearity == "tanh":
                scores = tf.tanh(self.scores)
            elif non_linearity == "sigmoid":
                scores = tf.sigmoid(scores)
            elif non_linearity == "softplus":
                scores = custom_softplus(scores)
            else:
                raise ValueError("Invalid non-linearity")

            return scores
            # with tf.Session(config=self.tf_config) as sess:
            #     sess.run(tf.global_variables_initializer())
            #     return sess.run(scores)
        else:
            dataset_handle = NumpyDatasetAdapter()
            dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)
            dataset_handle.set_data(X, "test", mapped_status=from_idx)

            self.eval_dataset_handle = dataset_handle

            # build tf graph for predictions
            self.rnd = check_random_state(self.seed)
            # tf.random.set_random_seed(self.seed)
            tf.random.set_seed(self.seed)
            # load the parameters
            # build the eval graph
            self._initialize_eval_graph()

            # with tf.Session(config=self.tf_config) as sess:
            #     sess.run(tf.tables_initializer())
            #     sess.run(tf.global_variables_initializer())

            try:
                # sess.run(self.set_training_false)
                self.set_training_false
            except AttributeError:
                pass

            scores = []

            for _ in tqdm(
                range(self.eval_dataset_handle.get_size("test")),
                disable=(not self.verbose),
            ):
                # score = sess.run(self.score_positive)
                score = self.score_positive
                scores.append(score)

            return scores

    def is_fitted_on(self, X):
        """Determine heuristically if a model was fitted on the given triples.

        :param X: The triples to score.
        :type X: ndarray, shape [n, 3]
        :return: True if the number of unique entities and relations in X
        :rtype: bool
        """

        if not self.is_fitted:
            msg = "Model has not been fitted."
            logger.error(msg)
            raise RuntimeError(msg)

        unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
        unique_rel = np.unique(X[:, 1])

        if not len(unique_ent) == len(self.ent_to_idx.keys()):
            return False
        elif not len(unique_rel) == len(self.rel_to_idx.keys()):
            return False

        return True

    def _calibrate_with_corruptions(self, X_pos, batches_count):
        """
        Calibrates model with corruptions. The corruptions are hard-coded to be subject and object ('s,o') with all
        available entities.

        :param X_pos: Numpy array of positive triples.
        :type X_pos: ndarray (shape [n, 3])
        :param batches_count: Number of batches to complete one epoch of the Platt scaling training.
        :type batches_count: int
        :return: scores_pos: Tensor with positive scores.
        scores_neg: Tensor with negative scores (generated by the corruptions).
        dataset_handle: Dataset handle (only used for clean-up).
        :rtype: tf.Tensor, tf.Tensor, NumpyDatasetAdapter
        """

        dataset_handle = NumpyDatasetAdapter()
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)

        dataset_handle.set_data(X_pos, "pos")

        gen_fn = partial(
            dataset_handle.get_next_batch,
            batches_count=batches_count,
            dataset_type="pos",
        )
        dataset = tf.data.Dataset.from_generator(
            gen_fn, output_types=tf.int32, output_shapes=(1, None, 3)
        )
        dataset = dataset.repeat().prefetch(1)
        dataset_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)

        x_pos_tf = dataset_iter.get_next()[0]

        e_s, e_p, e_o = self._lookup_embeddings(x_pos_tf)
        scores_pos = self._fn(e_s, e_p, e_o)

        x_neg_tf = generate_corruptions_for_fit(
            x_pos_tf,
            entities_list=None,
            eta=1,
            corrupt_side="s,o",
            entities_size=len(self.ent_to_idx),
            rnd=self.seed,
        )

        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)
        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)

        return scores_pos, scores_neg, dataset_handle

    def _calibrate_with_negatives(self, X_pos, X_neg):
        """Calibrates model with two datasets, one with positive triples and another with negative triples.

        :param X_pos: Numpy array of positive triples.
        :type X_pos: ndarray (shape [n, 3])
        :param X_neg: Numpy array of negative triples.
        :type X_neg: ndarray (shape [n, 3])
        :return: scores_pos: Tensor with positive scores.

        scores_neg: Tensor with negative scores.
        :rtype: tf.Tensor, tf.Tensor
        """

        x_neg = to_idx(X_neg, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_neg_tf = tf.Variable(x_neg, dtype=tf.int32, trainable=False)

        x_pos = to_idx(X_pos, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_pos_tf = tf.Variable(x_pos, dtype=tf.int32, trainable=False)

        e_s, e_p, e_o = self._lookup_embeddings(x_neg_tf)
        scores_neg = self._fn(e_s, e_p, e_o)

        e_s, e_p, e_o = self._lookup_embeddings(x_pos_tf)
        scores_pos = self._fn(e_s, e_p, e_o)

        return scores_pos, scores_neg

    def _calibrate(
        self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50
    ):
        """
        Calibrate predictions tod o: un-underscore this method later

        The method implements the heuristics described in :cite:`calibration`, using Platt scaling
        :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba` after calibration is done.

        Calibration should ideally be done on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. Both positive and negative triples are provided via ``X_pos`` and ``X_neg`` respectively. The optimization
        is done using a second-order method (limited-memory BFGS), therefore no hyperparameter needs to be specified.

        #. Only positive triples are provided, and the negative triples are generated by corruptions just like it is
        done in training or evaluation. The optimization is done using a first-order method (ADAM), therefore
        ``batches_count`` and ``epochs`` must be specified.


        The base rate of positive triples has a large influence on calibration. As a result, for mode (2) of
        operation, the user must supply the "positive base rate" option. For mode (1), the relative sizes of the
        positive and negative sets can deduce this automatically, but the user can override this by specifying a
        value to "positive base rate."

        When calibrating without negatives, the most difficult problem is defining the positive base rate. This is
        determined by the user's selection of which triples will be analyzed during testing. Take WN11 as an example:
        it has approximately 50% positive triples on both the validation and test sets, so the positive base rate is
        50%. However, if the user resamples it to have 75% positives and 25% negatives, the prior calibration will be
        impaired. The user must now recalibrate the model with a 75% positive base rate. As a result, this parameter
        is defined by how the user interacts with the dataset and cannot be determined mechanically or a priori.

        .. Note ::
            Incompatible with large graph mode (i.e. if ``self.dealing_with_large_graphs=True``).

        :param X_pos: Numpy array of positive triples.
        :type X_pos: ndarray (shape [n, 3])
        :param X_neg: Numpy array of negative triples.

            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.
        :type X_neg: ndarray (shape [n, 3])
        :param positive_base_rate: Base rate of positive statements.

            For example, if we assume that any inquiry has a fifty-fifty chance of being true, the base rate is 50%.

            If ``X_neg`` is provided and this is `None`, the relative sizes of ``X_pos`` and ``X_neg`` will be used
            to determine the base rate. For example, if we have 50 positive triples and 200 negative triples,
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
        >>> from emgraph.models import TransE
        >>>
        >>> X = BaseDataset.load_dataset(DatasetType.WN11)
        >>> X_valid_pos = X['valid'][X['valid_labels']]
        >>> X_valid_neg = X['valid'][~X['valid_labels']]
        >>>
        >>> model = TransE(batches_count=64, seed=0, epochs=500, k=100, eta=20,
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

        if not self.is_fitted:
            msg = "Model has not been fitted."
            logger.error(msg)
            raise RuntimeError(msg)

        if self.dealing_with_large_graphs:
            msg = "Calibration is incompatible with large graph mode."
            logger.error(msg)
            raise ValueError(msg)

        if positive_base_rate is not None and (
            positive_base_rate <= 0 or positive_base_rate >= 1
        ):
            msg = "positive_base_rate must be a value between 0 and 1."
            logger.error(msg)
            raise ValueError(msg)

        dataset_handle = None

        try:
            # tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            # tf.random.set_random_seed(self.seed)
            tf.random.set_seed(self.seed)

            self._load_model_from_trained_params()

            if X_neg is not None:
                if positive_base_rate is None:
                    positive_base_rate = len(X_pos) / (len(X_pos) + len(X_neg))
                scores_pos, scores_neg = self._calibrate_with_negatives(X_pos, X_neg)
            else:
                if positive_base_rate is None:
                    msg = (
                        "When calibrating with randomly generated negative corruptions, "
                        "`positive_base_rate` must be set to a value between 0 and 1."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                (
                    scores_pos,
                    scores_neg,
                    dataset_handle,
                ) = self._calibrate_with_corruptions(X_pos, batches_count)

            n_pos = len(X_pos)
            n_neg = len(X_neg) if X_neg is not None else n_pos

            scores_tf = tf.concat([scores_pos, scores_neg], axis=0)
            labels = tf.concat(
                [
                    tf.cast(
                        tf.fill(tf.shape(scores_pos), (n_pos + 1.0) / (n_pos + 2.0)),
                        tf.float32,
                    ),
                    tf.cast(
                        tf.fill(tf.shape(scores_neg), 1 / (n_neg + 2.0)), tf.float32
                    ),
                ],
                axis=0,
            )

            # Platt scaling model
            # w = tf.get_variable('w', initializer=0.0, dtype=tf.float32)
            # b = tf.get_variable('b', initializer=np.log((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32),
            #                     dtype=tf.float32)

            # w = tf.Variable(tf.constant_initializer(0.0, shape=[scores_tf.shape]), name='w', dtype=tf.float32)
            w = make_variable(
                name="w",
                shape=scores_tf.shape,
                initializer=tf.zeros_initializer(),
                dtype=tf.float32,
            )
            # w = self._make_variable(name='w', shape=tf.TensorShape(None), initializer=tf.zeros_initializer(), dtype=tf.float32)
            print(
                "np.log((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32): ",
                tf.constant_initializer(
                    np.log((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32)
                ),
            )
            b = make_variable(
                name="b",
                shape=scores_tf.shape,
                initializer=tf.constant_initializer(
                    np.log((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32)
                ),
                dtype=tf.float32,
            )

            print(
                f"w: {w}\ntf.stop_gradient(scores_tf): {tf.stop_gradient(scores_tf)}\nb: {b}"
            )
            # logits = -(w * tf.stop_gradient(scores_tf) + b)
            logits = -(w * scores_tf + b)

            # Sample weights make sure the given positive_base_rate will be achieved irrespective of batch sizes
            weigths_pos = tf.size(scores_neg) / tf.size(scores_pos)
            weights_neg = (1.0 - positive_base_rate) / positive_base_rate
            weights = tf.concat(
                [
                    tf.cast(tf.fill(tf.shape(scores_pos), weigths_pos), tf.float32),
                    tf.cast(tf.fill(tf.shape(scores_neg), weights_neg), tf.float32),
                ],
                axis=0,
            )

            print("w: ", w, "\nweights: ", weights)

            # loss = functools.partial(tf.compat.v1.losses.sigmoid_cross_entropy, labels, logits, weights=weights)
            loss = functools.partial(
                tf.nn.sigmoid_cross_entropy_with_logits, labels, logits
            )

            # optimizer = tf.train.AdamOptimizer()
            optimizer = tf.keras.optimizers.Adam()

            train = optimizer.minimize(loss, [logits, labels])

            # with tf.Session(config=self.tf_config) as sess:
            #     sess.run(tf.global_variables_initializer())

            epoch_iterator_with_progress = tqdm(
                range(1, epochs + 1), disable=(not self.verbose), unit="epoch"
            )
            for _ in epoch_iterator_with_progress:
                losses = []
                for batch in range(batches_count):
                    # loss_batch, _ = sess.run([loss, train])
                    loss_batch, _ = [loss, train]
                    losses.append(loss_batch)
                if self.verbose:
                    msg = "Calibration Loss: {:10f}".format(sum(losses) / batches_count)
                    logger.debug(msg)
                    epoch_iterator_with_progress.set_description(msg)

            # self.calibration_parameters = sess.run([w, b])
            self.calibration_parameters = [w, b]
            self.is_calibrated = True
        finally:
            if dataset_handle is not None:
                dataset_handle.cleanup()

    def _predict_proba(self, X):
        """Predicts probabilities using the Platt scaling model (after calibration).

        Model must be calibrated beforehand with the ``calibrate`` method.

        :param X: Numpy array of triples to be evaluated.
        :type X: ndarray, shape [n, 3]
        :return: Probability of each triple to be true according to the Platt scaling calibration.
        :rtype: ndarray, shape [n, 3]
        """

        if not self.is_calibrated:
            msg = "Model has not been calibrated. Please call `model.calibrate(...)` before predicting probabilities."
            logger.error(msg)
            raise RuntimeError(msg)

        # tf.reset_default_graph()

        self._load_model_from_trained_params()

        w = tf.Variable(
            self.calibration_parameters[0], dtype=tf.float32, trainable=False
        )
        b = tf.Variable(
            self.calibration_parameters[1], dtype=tf.float32, trainable=False
        )

        x_idx = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_tf = tf.Variable(x_idx, dtype=tf.int32, trainable=False)

        e_s, e_p, e_o = self._lookup_embeddings(x_tf)
        scores = self._fn(e_s, e_p, e_o)
        logits = -(w * scores + b)
        probas = tf.sigmoid(logits)

        # with tf.Session(config=self.tf_config) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     return sess.run(probas)
        return probas
