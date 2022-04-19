import abc


class EmgraphBaseDatasetAdaptor(abc.ABC):
    """An abstract class that provides the infrastructure for defining new data adaptors and pipelines to feed Emgraph.

    It delivers the standard interface for adding additional data sources. The following methods are the main
    components for this class:
        - :func:`use_mappings`: Activate provided mapping for the datasource.
        - :func:`generate_mappings`: Generate the mappings from the training set.
        - :func:`get_size`: Return the size of the dataset
        - :func:`data_exists`: Check if the provided dataset type (train, test, valid, etc.) exists.
        - :func:`set_data`: Set the dataset based on the specified dataset_type.
        - :func:`map_data`: Map the data to the ent_to_idx and rel_to_idx mappings.
        - :func:`set_filter`: Set filters while generating evaluation batches.
        - :func:`get_next_batch`: Generate the next batch of data.
        - :func:`cleanup`: Clean up the internal state.

    :Authors:
        `Soran ghaderi <https://soran-ghaderi.github.io/>`_,
        Taleb Zarhesh
    """

    def __init__(self):
        """
        Initialize the class variables

        """
        self.dataset = {}

        # relation to idx mappings
        self.rel_to_idx = {}
        # entities to idx mappings
        self.ent_to_idx = {}
        # Mapped status of each dataset
        self.mapped_status = {}
        # link weights for focusE
        self.focusE_numeric_edge_values = {}

    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Activate provided mapping for the datasource.

        :param rel_to_idx: Relation to idx mapping
        :type rel_to_idx: dict
        :param ent_to_idx: entity to idx mapping
        :type ent_to_idx: dict
        :return:
        :rtype:
        """

        self.rel_to_idx = rel_to_idx
        self.ent_to_idx = ent_to_idx
        # set the mapped status to false, since we are changing the dictionary
        for key in self.dataset.keys():
            self.mapped_status[key] = False

    def generate_mappings(self, use_all=False):
        """Generate the mappings from the training set. If "use_all==True" the function will use the whole dataset.

        :param use_all: Whether to use the whole dataset for "mappings" generation (default=False)
        :type use_all: bool
        :return:
            - rel_to_idx: Relation to idx
            - ent_to_idx: Entity to idx
        :rtype: dict, dict
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def get_size(self, dataset_type="train"):
        """Return the size of the dataset

        :param dataset_type: Dataset type (default='train')
        :type dataset_type: str
        :return: Dataset size
        :rtype: int
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def data_exists(self, dataset_type="train"):
        """Check if the provided dataset type (train, test, valid, etc.) exists.

        :param dataset_type: Type of the dataset (default='train')
        :type dataset_type: str
        :return: True if the specified dataset type exists; False otherwise
        :rtype: bool
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        """Set the dataset based on the specified dataset_type.

        :param dataset: Dataset of triples
        :type dataset: str
        :param dataset_type: The type of the dataset if the 'dataset' is an ND-Array (default=None)
        :type dataset_type: str
        :param mapped_status: Whether the dataset is mapped to indices or not (default=False)
        :type mapped_status: bool
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def map_data(self, remap=False):
        """Map the data to the ent_to_idx and rel_to_idx mappings.

        :param remap: Remap the data (used after the dictionaries get updated) (default=False)
        :type remap: bool
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def set_filter(self, filter_triples):
        """Set filters while generating evaluation batches.

        :param filter_triples: Filter triples
        :type filter_triples: nd-array
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
        """Generate the next batch of data.

        :param batches_count:  Number of batches per epoch (defaults to -1 means batch size of 1) (default=-1)
        :type batches_count: int
        :param dataset_type: Type of the dataset (default='train')
        :type dataset_type: str
        :param use_filter: Whether to return the filters' metadata (default=False)
        :type use_filter: bool
        :return:
            - batch_output: yields a batch of triples from the dataset type specified.
            - participating_objects: all objects that were involved in the s-p-? relation. This is returned only if use_filter is set to true.
            - participating_subjects: all subjects that were involved in the ?-p-o relation. This is returned only if use_filter is set to true.
        :rtype: nd-array, nd-array [n,1], nd-array [n,1]
        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def cleanup(self):
        """Clean up the internal state.

        :raises: NotImplementedError if the method is not overridden
        """

        raise NotImplementedError('Abstract Method not implemented!')
