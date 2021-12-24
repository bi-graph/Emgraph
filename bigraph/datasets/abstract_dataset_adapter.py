

import abc

class BigraphDatasetAdapter(abc.ABC):
    """
    Abstract class for dataset adapters Developers can use this abstract class for adapting and creating a pipeline of
    data to feed Bigraph.

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
        """
        Use an existing mapping with the datasource.

        """
        self.rel_to_idx = rel_to_idx
        self.ent_to_idx = ent_to_idx
        # set the mapped status to false, since we are changing the dictionary
        for key in self.dataset.keys():
            self.mapped_status[key] = False


    def generate_mappings(self, use_all=False):
        """
        Generate the mappings from the training set. If "use_all==True" the function will use the whole dataset.

        :param use_all: Whether to use the whole dataset for "mappings" generation
        :type use_all: bool
        :return: Two dictionaries: relation to idx and entity to idx respectively
        :rtype: dict
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def get_size(self, dataset_type="train"):
        """
        Size of the dataset

        :param dataset_type: Dataset type
        :type dataset_type: str
        :return: Size of the dataset
        :rtype: int
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def data_exists(self, dataset_type="train"):
        """
        Check if the dataset type (train, test, valid, etc.) exists.

        :param dataset_type: Type of the dataset
        :type dataset_type: str
        :return: True if the specified dataset type exists; False otherwise
        :rtype: bool
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        """
        Set the dataset based on the specified dataset_type.

        :param dataset: Dataset of triples
        :type dataset: str
        :param dataset_type: The type of the dataset if the 'dataset' is an ND-Array
        :type dataset_type: str
        :param mapped_status: Whether the dataset is mapped to indices or not
        :type mapped_status: bool
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def map_data(self, remap=False):
        """
        Map the data to the ent_to_idx and rel_to_idx mappings.

        :param remap: Remap the data (used after the dictionaries get updated)
        :type remap: bool
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def set_filter(self, filter_triples):
        """
        Set filters while generating evaluation batches.

        :param filter_triples: Filter triples
        :type filter_triples: nd-array
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
        """
        Generate the next batch of data.

        :param batches_count:  Number of batches per epoch (defaults to -1 means batch size of 1)
        :type batches_count: int
        :param dataset_type: Type of the dataset
        :type dataset_type: str
        :param use_filter: Whether to return the filters' metadata
        :type use_filter: bool
        :return: batch_output: yields a batch of triples from the dataset type specified.
        participating_objects: all objects that were involved in the s-p-? relation. This is returned only
            if use_filter is set to true.
        participating_subjects: all subjects that were involved in the ?-p-o relation. This is returned only
            if use_filter is set to true.
        :rtype: nd-array
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def cleanup(self):
        """
        Clean up the internal state.

        """

        raise NotImplementedError('Abstract Method not implemented!')
