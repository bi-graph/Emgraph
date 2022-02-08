
import numpy as np
from ..datasets import NumpyDatasetAdapter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OneToNDatasetAdapter(NumpyDatasetAdapter):
    r"""1-to-N Dataset Adapter.

        Given a triples dataset X comprised of n triples in the form (s, p, o), this dataset adapter will
        generate one-hot outputs for each (s, p) tuple to all entities o that are found in X.

        E.g: X = [[a, p, b],
                  [a, p, d],
                  [c, p, d],
                  [c, p, e],
                  [c, p, f]]

        Gives a one-hot vector mapping of entities to indices:

            Entities: [a, b, c, d, e, f]
            Indices: [0, 1, 2, 3, 4, 5]

        One-hot outputs are produced for each (s, p) tuple to all valid object indices in the dataset:

                  #  [a, b, c, d, e, f]
            (a, p) : [0, 1, 0, 1, 0, 0]

        The ```get_next_batch``` function yields the (s, p, o) triple and one-hot vector corresponding to the (s, p)
        tuple.

        If batches are generated with ```unique_pairs=True``` then only one instance of each unique (s, p) tuple
        is returned:

            (a, p) : [0, 1, 0, 1, 0, 0]
            (c, p) : [0, 0, 0, 1, 1, 1]

        Otherwise batch outputs are generated in dataset order (required for evaluating test set, but gives a higher
        weight to more frequent (s, p) pairs if used during model training):

            (a, p) : [0, 1, 0, 1, 0, 0]
            (a, p) : [0, 1, 0, 1, 0, 0]
            (c, p) : [0, 0, 0, 1, 1, 1]
            (c, p) : [0, 0, 0, 1, 1, 1]
            (c, p) : [0, 0, 0, 1, 1, 1]

    """

    def __init__(self, low_memory=False):
        """Initialize the OneToNDatasetAdapter variables.

        :param low_memory: Flag to use low-memory mode. If True: The output vectors will be generated in the batch-yield
        function, which lowers the memory usage but increases the training time.
        :type low_memory: bool
        """

        super(OneToNDatasetAdapter, self).__init__()

        self.filter_mapping = None
        self.filtered_status = {}
        self.paired_status = {}
        self.output_mapping = None
        self.output_onehot = {}
        self.low_memory = low_memory


