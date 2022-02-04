
import numpy as np
from ..datasets import BigraphDatasetAdapter, SQLiteAdapter


class NumpyDatasetAdapter(BigraphDatasetAdapter):

    def __init__(self):
        """Initialize the NumpyDatasetAdapter variables
        """

        super(NumpyDatasetAdapter, self).__init__()
        # NumpyDatasetAdapter uses SQLAdapter to filter (if filters are set)
        self.filter_adapter = None


    def generate_mappings(self, use_all=False):
        """Generate mappings from either train set or use all dataset to generate mappings.

        :param use_all: Whether to use all the data or not. If True it uses all the data else the train
        set (default: False)
        :type use_all: bool
        :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping: entity to idx mapping
        :rtype: dict, dict
        """

        from ..evaluation import create_mappings
        if use_all:
            complete_dataset = []
            for key in self.dataset.keys():
                complete_dataset.append(self.dataset[key])
            self.rel_to_idx, self.ent_to_idx = create_mappings(np.concatenate(complete_dataset, axis=0))

        else:
            self.rel_to_idx, self.ent_to_idx = create_mappings(self.dataset["train"])

        return self.rel_to_idx, self.ent_to_idx