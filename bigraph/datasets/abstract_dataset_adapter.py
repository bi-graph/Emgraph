

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

        raise NotImplementedError('Abstract Method not implemented!')

    def get_size(self, dataset_type="train"):

        raise NotImplementedError('Abstract Method not implemented!')

    def data_exists(self, dataset_type="train"):

        raise NotImplementedError('Abstract Method not implemented!')

    def set_data(self, dataset, dataset_type=None, mapped_status=False):

        raise NotImplementedError('Abstract Method not implemented!')

    def map_data(self, remap=False):

        raise NotImplementedError('Abstract Method not implemented!')

    def set_filter(self, filter_triples):

        raise NotImplementedError('Abstract Method not implemented!')

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):

        raise NotImplementedError('Abstract Method not implemented!')

    def cleanup(self):

        raise NotImplementedError('Abstract Method not implemented!')
