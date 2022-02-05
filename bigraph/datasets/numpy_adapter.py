
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

    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Use an existing mapping with the datasource.

        :param rel_to_idx: Relation to idx mapping
        :type rel_to_idx: dict
        :param ent_to_idx: entity to idx mapping
        :type ent_to_idx: dict
        :return: -
        :rtype: -
        """

        super().use_mappings(rel_to_idx, ent_to_idx)

    def get_size(self, dataset_type="train"):
        """Return the size of the specified dataset

        :param dataset_type: Dataset type
        :type dataset_type: str
        :return: Size of the specified dataset
        :rtype: int
        """

        return self.dataset[dataset_type].shape[0]

    def data_exists(self, dataset_type="train"):
        """Checks if a dataset_type exists in the adapter.

        :param dataset_type: Dataset type
        :type dataset_type: str
        :return: True if exists, False otherwise
        :rtype: bool
        """

        return dataset_type in self.dataset.keys()