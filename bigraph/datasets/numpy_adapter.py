
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

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
        """Generate the next batch of data.

        :param batches_count: Number of batches per epoch (defaults to -1 means batch size of 1)
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
        :rtype: nd-array, nd-array [n,1], nd-array [n,1]
        """

        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if batches_count == -1:
            batch_size = 1
            batches_count = self.get_size(dataset_type)
        else:
            batch_size = int(np.ceil(self.get_size(dataset_type) / batches_count))

        for i in range(batches_count):
            output = []
            out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
            output.append(out)

            try:
                focusE_numeric_edge_values_batch = self.focusE_numeric_edge_values[
                                                       dataset_type][(i * batch_size):((i + 1) * batch_size), :]
                output.append(focusE_numeric_edge_values_batch)
            except KeyError:
                pass

            if use_filter:
                # get the filter values by querying the database
                participating_objects, participating_subjects = self.filter_adapter.get_participating_entities(out)
                output.append(participating_objects)
                output.append(participating_subjects)

            yield output