
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