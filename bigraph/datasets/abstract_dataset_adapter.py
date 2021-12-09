

import abc

class BigraphDatasetAdapter(abc.ABC):
    """Abstract class for dataset adapters
       Developers can use this abstract class for adapting and creating a pipeline of data to feed Bigraph.
    """

    def __init__(self):
        """Initialize the class variables
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
        """Use an existing mapping with the datasource.
        """
        self.rel_to_idx = rel_to_idx
        self.ent_to_idx = ent_to_idx
        # set the mapped status to false, since we are changing the dictionary
        for key in self.dataset.keys():
            self.mapped_status[key] = False