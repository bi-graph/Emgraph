
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

        self.set_data(filter_triples, 'filter', mapped_status)
        self.filter_mapping = self.generate_output_mapping('filter')

    def generate_outputs(self, dataset_type='train', use_filter=False, unique_pairs=True):
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
            msg = 'Unable to generate outputs: dataset `{}` not found. ' \
                  'Use `set_data` to set dataset in adapter first.'.format(dataset_type)
            raise KeyError(msg)

        if dataset_type in ['valid', 'test']:
            if unique_pairs:
                # This is just a friendly warning - in most cases the test and valid sets should NOT be unique_pairs.
                msg = 'Generating outputs for dataset `{}` with unique_pairs=True. ' \
                      'Are you sure this is desired behaviour?'.format(dataset_type)
                logger.warning(msg)

        if use_filter:
            if self.filter_mapping is None:
                msg = 'Filter not found: cannot generate one-hot outputs with `use_filter=True` ' \
                      'if a filter has not been set.'
                raise ValueError(msg)
            else:
                output_dict = self.filter_mapping
        else:
            if self.output_mapping is None:
                msg = 'Output mapping was not created before generating one-hot vectors. '
                raise ValueError(msg)
            else:
                output_dict = self.output_mapping

        if self.low_memory:
            # With low_memory=True the output indices are generated on the fly in the batch yield function
            pass
        else:
            if unique_pairs:
                X = np.unique(self.dataset[dataset_type][:, [0, 1]], axis=0).astype(np.int32)
            else:
                X = self.dataset[dataset_type]

            # Initialize np.array of shape [len(X), num_entities]
            self.output_onehot[dataset_type] = np.zeros((len(X), len(self.ent_to_idx)), dtype=np.int8)

            # Set one-hot indices using output_dict
            for i, x in enumerate(X):
                indices = output_dict.get((x[0], x[1]), [])
                self.output_onehot[dataset_type][i, indices] = 1

            # Set flags indicating filter and unique pair status of outputs for given dataset.
            self.filtered_status[dataset_type] = use_filter
            self.paired_status[dataset_type] = unique_pairs

    def generate_output_mapping(self, dataset_type='train'):
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
            logger.debug('Dataset {} is in adapter, but filtered_status is not set.'.format(dataset_type))
            return False

        if dataset_type not in self.paired_status.keys():
            logger.debug('Dataset {} is in adapter, but paired_status is not set.'.format(dataset_type))
            return False

        if use_filter != self.filtered_status[dataset_type]:
            return False

        if unique_pairs != self.paired_status[dataset_type]:
            return False

        return True