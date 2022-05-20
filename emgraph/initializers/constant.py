import tensorflow as tf

from emgraph.initializers.initializer import Initializer
from emgraph.initializers.utils import export_emgraph_initializer


@export_emgraph_initializer("constant", ["entity", "relation"])
class Constant(Initializer):
    """
    Initialize based on the specified constant values by the user.

    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, initializer_params={}, verbose=True, seed=0):
        """
        Initializes based on the specified constant values by the user.

        :param initializer_params: Key-value pairs. The initializer gets the params from the keys:
            - **entity**: (np.ndarray.float32). Initialize entity embeddings
            - **relation**: (np.ndarray.float32). Initialize relation embeddings
        :type initializer_params: dict
        :param verbose: Activate verbose
        :type verbose: bool
        :param seed: Random state for random number generator
        :type seed: int
        """

        super(Constant, self).__init__(initializer_params, verbose, seed)

    def _init_hyperparams(self, hyperparam_dict):
        """
        Initialize the hyperparameters.

        :param hyperparam_dict: Key-value pairs. The initializer gets the params from the keys
        :type hyperparam_dict: dict
        """

        try:
            self._initializer_params['entity'] = hyperparam_dict['entity']
            self._initializer_params['relation'] = hyperparam_dict['relation']
        except KeyError:
            raise Exception(
                'Initial values of both entity and relation embeddings need to '
                'be passed to the initializer!'
                )
        if self.verbose:
            self._display_params()

    def _get_tf_initializer(self, in_shape=None, out_shape=None, concept='e'):
        """
        Generate an initialized Tensorflow node for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initializer instance
        :rtype: Initializer
        """

        if concept == 'e':
            assert self._initializer_params['entity'].shape[0] == in_shape and \
                   self._initializer_params['entity'].shape[1] == out_shape, \
                "Invalid shape for entity initializer!"
            return tf.constant_initializer(self._initializer_params['entity'], dtype=tf.float32)
        else:
            assert self._initializer_params['relation'].shape[0] == in_shape and \
                   self._initializer_params['relation'].shape[1] == out_shape, \
                "Invalid shape for relation initializer!"

            return tf.constant_initializer(self._initializer_params['relation'], dtype=tf.float32)

    def _get_np_initializer(self, in_shape, out_shape, concept='e'):
        """
        Generate an initialized Numpy array for the initializer.

        :param in_shape: Number of the layer's inputs.
        :type in_shape: int
        :param out_shape: Number of the layer's output.
        :type out_shape: int
        :param concept: Concept type (e: entity, r: relation)
        :type concept: str
        :return: Initialized weights (uniform distribution)
        :rtype: nd-array
        """

        if concept == 'e':
            assert self._initializer_params['entity'].shape[0] == in_shape and \
                   self._initializer_params['entity'].shape[1] == out_shape, \
                "Invalid shape for entity initializer!"

            return self._initializer_params['entity']
        else:
            assert self._initializer_params['relation'].shape[0] == in_shape and \
                   self._initializer_params['relation'].shape[1] == out_shape, \
                "Invalid shape for relation initializer!"

            return self._initializer_params['relation']
