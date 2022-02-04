
import numpy as np
from ..datasets import BigraphDatasetAdapter, SQLiteAdapter


class NumpyDatasetAdapter(BigraphDatasetAdapter):

    def __init__(self):
        """Initialize the NumpyDatasetAdapter variables
        """

        super(NumpyDatasetAdapter, self).__init__()
        # NumpyDatasetAdapter uses SQLAdapter to filter (if filters are set)
        self.filter_adapter = None