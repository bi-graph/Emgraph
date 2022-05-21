import logging

import numpy as np

from ..datasets import NumpyDatasetAdapter

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

    def set_filter(self, filter_triples, mapped_status=False):
        """Set filters for generating the evaluation batch.
            Note: This adapter uses SQL backend for filtering

        :param filter_triples: Filtering triples
        :type filter_triples: nd-array
        :param mapped_status: Whether the dataset is mapped to the indices
        :type mapped_status: bool
        :return: -
        :rtype: -
        """

        self.set_data(filter_triples, "filter", mapped_status)
        self.filter_mapping = self.generate_output_mapping("filter")

    def generate_outputs(
        self, dataset_type="train", use_filter=False, unique_pairs=True
    ):
        """Generate one-hot outputs for the specified dataset.

        :param dataset_type: Dataset type
        :type dataset_type: str
        :param use_filter: Whether to generate outputs using the filter set by `set_filter()`. Default: False
        :type use_filter: bool
        :param unique_pairs: Whether to generate outputs according to unique pairs of (subject, predicate), otherwise
            will generate outputs in same row-order as the triples in the specified dataset. Default: True.
        :type unique_pairs: bool
        :return: -
        :rtype: -
        """

        if dataset_type not in self.dataset.keys():
            msg = (
                "Unable to generate outputs: dataset `{}` not found. "
                "Use `set_data` to set dataset in adapter first.".format(dataset_type)
            )
            raise KeyError(msg)

        if dataset_type in ["valid", "test"]:
            if unique_pairs:
                # This is just a friendly warning - in most cases the test and valid sets should NOT be unique_pairs.
                msg = (
                    "Generating outputs for dataset `{}` with unique_pairs=True. "
                    "Are you sure this is desired behaviour?".format(dataset_type)
                )
                logger.warning(msg)

        if use_filter:
            if self.filter_mapping is None:
                msg = (
                    "Filter not found: cannot generate one-hot outputs with `use_filter=True` "
                    "if a filter has not been set."
                )
                raise ValueError(msg)
            else:
                output_dict = self.filter_mapping
        else:
            if self.output_mapping is None:
                msg = (
                    "Output mapping was not created before generating one-hot vectors. "
                )
                raise ValueError(msg)
            else:
                output_dict = self.output_mapping

        if self.low_memory:
            # With low_memory=True the output indices are generated on the fly in the batch yield function
            pass
        else:
            if unique_pairs:
                X = np.unique(self.dataset[dataset_type][:, [0, 1]], axis=0).astype(
                    np.int32
                )
            else:
                X = self.dataset[dataset_type]

            # Initialize np.array of shape [len(X), num_entities]
            self.output_onehot[dataset_type] = np.zeros(
                (len(X), len(self.ent_to_idx)), dtype=np.int8
            )

            # Set one-hot indices using output_dict
            for i, x in enumerate(X):
                indices = output_dict.get((x[0], x[1]), [])
                self.output_onehot[dataset_type][i, indices] = 1

            # Set flags indicating filter and unique pair status of outputs for given dataset.
            self.filtered_status[dataset_type] = use_filter
            self.paired_status[dataset_type] = unique_pairs

    def generate_output_mapping(self, dataset_type="train"):
        """Create dictionary keyed on (subject, predicate) to list of objects

        :param dataset_type: Dataset type
        :type dataset_type: str
        :return: Dictionary of mapped data
        :rtype: dict
        """

        # if data is not already mapped, then map before creating output map
        if not self.mapped_status[dataset_type]:
            self.map_data()

        output_mapping = dict()

        for s, p, o in self.dataset[dataset_type]:
            output_mapping.setdefault((s, p), []).append(o)

        return output_mapping

    def set_output_mapping(self, output_dict, clear_outputs=True):
        """Set the mapping used to generate one-hot outputs vectors.

        Setting a new output mapping will clear_outputs any previously generated outputs, as otherwise
        can lead to a situation where old outputs are returned from batch function.

        :param output_dict: Object indices of (subject, predicate)s
        :type output_dict: dict
        :param clear_outputs: Clears any one hot outputs held by the adapter, as otherwise can lead to a situation where onehot
            outputs generated by a different mapping are returned from the batch function. Default: True.
        :type clear_outputs: bool
        :return: -
        :rtype: -
        """

        self.output_mapping = output_dict

        # Clear any onehot outputs previously generated
        if clear_outputs:
            self.clear_outputs()

    def clear_outputs(self, dataset_type=None):
        """Clear the internal memory containing generated one-hot outputs.

        :param dataset_type: Dataset type to clear its outputs. (Default: None (clear all))
        :type dataset_type:
        :return:
        :rtype:
        """

        if dataset_type is None:
            self.output_onehot = {}
            self.filtered_status = {}
            self.paired_status = {}
        else:
            del self.output_onehot[dataset_type]
            del self.filtered_status[dataset_type]
            del self.paired_status[dataset_type]

    def verify_outputs(self, dataset_type, use_filter, unique_pairs):
        """Verify the correspondence of the generated outputs and specified filters and unique pairs.

        :param dataset_type: Dataset type
        :type dataset_type: str
        :param use_filter: Whether to generate one-hot outputs from filtered or not-filtered datasets
        :type use_filter: bool
        :param unique_pairs: Whether to generate one-hot outputs based on the unique (s, p) pairs or dataset order
        :type unique_pairs: bool
        :return: False if outputs need to be regenerated for the specified dataset and parameters, otherwise True
        :rtype: bool
        """

        if dataset_type not in self.output_onehot.keys():
            # One-hot outputs have not been generated for this dataset_type
            return False

        if dataset_type not in self.filtered_status.keys():
            # This shouldn't happen.
            logger.debug(
                "Dataset {} is in adapter, but filtered_status is not set.".format(
                    dataset_type
                )
            )
            return False

        if dataset_type not in self.paired_status.keys():
            logger.debug(
                "Dataset {} is in adapter, but paired_status is not set.".format(
                    dataset_type
                )
            )
            return False

        if use_filter != self.filtered_status[dataset_type]:
            return False

        if unique_pairs != self.paired_status[dataset_type]:
            return False

        return True

    def get_next_batch(
        self,
        batches_count=-1,
        dataset_type="train",
        use_filter=False,
        unique_pairs=True,
    ):
        """Generate the next batch of data.

        :param batches_count: Number of batches per epoch (defaults to -1 means batch size of 1)
        :type batches_count: int
        :param dataset_type: Type of the dataset
        :type dataset_type: str
        :param use_filter: Flag whether to generate the one-hot outputs from filtered or not-filtered dataset
        :type use_filter: bool
        :param unique_pairs: Flag Whether to generate one-hot outputs based on the unique (s, p) pairs or dataset order
        :type unique_pairs: bool
        :return: Batch_output: A batch of triples from the specified dataset type. If unique_pairs=True, then the
        object column will be set to zeros.
        batch_onehot: A batch of onehot arrays corresponding to `batch_output` triples
        :rtype: nd-array, shape=[batch_size, 3], nd-array
        """

        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if unique_pairs:
            X = np.unique(self.dataset[dataset_type][:, [0, 1]], axis=0).astype(
                np.int32
            )
            X = np.c_[X, np.zeros(len(X))]  # Append dummy object columns
        else:
            X = self.dataset[dataset_type]
        dataset_size = len(X)

        if batches_count == -1:
            batch_size = 1
            batches_count = dataset_size
        else:
            batch_size = int(np.ceil(dataset_size / batches_count))

        if use_filter and self.filter_mapping is None:
            msg = "Cannot set `use_filter=True` if a filter has not been set in the adapter. "
            logger.error(msg)
            raise ValueError(msg)

        if not self.low_memory:

            if not self.verify_outputs(
                dataset_type, use_filter=use_filter, unique_pairs=unique_pairs
            ):
                # Verifies that onehot outputs are as expected given filter and unique_pair settings
                msg = "Generating one-hot outputs for {} [filtered: {}, unique_pairs: {}]".format(
                    dataset_type, use_filter, unique_pairs
                )
                logger.info(msg)
                self.generate_outputs(
                    dataset_type, use_filter=use_filter, unique_pairs=unique_pairs
                )

            # Yield batches
            for i in range(batches_count):
                out = np.int32(X[(i * batch_size) : ((i + 1) * batch_size), :])
                out_onehot = self.output_onehot[dataset_type][
                    (i * batch_size) : ((i + 1) * batch_size), :
                ]

                yield out, out_onehot

        else:
            # Low-memory, generate one-hot outputs per batch on the fly
            if use_filter:
                output_dict = self.filter_mapping
            else:
                output_dict = self.output_mapping

            # Yield batches
            for i in range(batches_count):

                out = np.int32(X[(i * batch_size) : ((i + 1) * batch_size), :])
                out_onehot = np.zeros(
                    shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int32
                )

                for j, x in enumerate(out):
                    indices = output_dict.get((x[0], x[1]), [])
                    out_onehot[j, indices] = 1

                yield out, out_onehot

    def get_next_batch_subject_corruptions(
        self, batch_size=-1, dataset_type="train", use_filter=True
    ):
        """Batch generator for subject corruptions.

        To avoid multiple redundant forward-passes through the network, subject corruptions are performed once for
        each relation, and results accumulated for valid test triples.

        If there are no test triples for a relation, then that relation is ignored.

        Use batch_size to control memory usage (as a batch_size*N tensor will be allocated, where N is number
        of unique entities.)

        :param batch_size: Maximum size of the returned batch
        :type batch_size: int
        :param dataset_type: Dataset type
        :type dataset_type: str
        :param use_filter: Flag whether to generate the one-hot outputs from filtered or not-filtered dataset
        :type use_filter: bool
        :return: test_triples: The set of all triples from the dataset type specified that include the
        predicate currently returned in batch_triples.
        batch_triples: A batch of triples corresponding to subject corruptions of just one predicate.
        batch_onehot: A batch of onehot arrays corresponding to the batch_triples output.
        :rtype: nd-array of shape (?, 3), nd-array of shape (M, 3) where M is the subject corruption batch size,
        nd-array of shape (M, N), where N is number of unique entities
        """

        if use_filter:
            output_dict = self.filter_mapping
        else:
            output_dict = self.output_mapping

        if batch_size == -1:
            batch_size = len(self.ent_to_idx)

        ent_list = np.array(list(self.ent_to_idx.values()))
        rel_list = np.array(list(self.rel_to_idx.values()))

        for rel in rel_list:

            # Select test triples that have this relation
            rel_idx = self.dataset[dataset_type][:, 1] == rel
            test_triples = self.dataset[dataset_type][rel_idx]

            ent_idx = 0

            while ent_idx < len(ent_list):

                ents = ent_list[ent_idx : ent_idx + batch_size]
                ent_idx += batch_size

                # Note: the object column is just a dummy value so set to 0
                out = np.stack(
                    [ents, np.repeat(rel, len(ents)), np.repeat(0, len(ents))], axis=1
                )

                # Set one-hot filter
                out_filter = np.zeros((out.shape[0], len(ent_list)), dtype=np.int8)
                for j, x in enumerate(out):
                    indices = output_dict.get((x[0], x[1]), [])
                    out_filter[j, indices] = 1

                yield test_triples, out, out_filter

    def _validate_data(self, data):
        """Validates the data.

        :param data: Data to be validated
        :type data: np.ndarray
        :return: -
        :rtype: -
        """

        if type(data) != np.ndarray:
            msg = "Invalid type for input data. Expected ndarray, got {}".format(
                type(data)
            )
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = "Invalid size for input data. Expected number of column 3, got {}".format(
                np.shape(data)[1]
            )
            raise ValueError(msg)

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        """Set the dataset based on the type.
        Note: If you pass the same dataset type (which exists) it will be overwritten

        :param dataset: Dataset of triples
        :type dataset: dict or nd-array
        :param dataset_type: If dataset == nd-array then indicates the type of the data
        :type dataset_type: str
        :param mapped_status: Whether the dataset is mapped to the indices
        :type mapped_status: bool
        :return: -
        :rtype: -
        """

        if isinstance(dataset, dict):
            for key in dataset.keys():
                self._validate_data(dataset[key])
                self.dataset[key] = dataset[key]
                self.mapped_status[key] = mapped_status
        elif dataset_type is not None:
            self._validate_data(dataset)
            self.dataset[dataset_type] = dataset
            self.mapped_status[dataset_type] = mapped_status
        else:
            raise Exception(
                "Incorrect usage. Expected a dictionary or a combination of dataset and it's type."
            )

        # If the concept-idx mappings are present, then map the passed dataset
        if not (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0):
            print("Mapping set data: {}".format(dataset_type))
            self.map_data()
