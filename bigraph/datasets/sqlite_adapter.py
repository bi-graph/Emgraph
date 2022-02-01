
import numpy as np
from ..datasets import BigraphDatasetAdapter
import tempfile
import sqlite3
import time
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SQLiteAdapter(BigraphDatasetAdapter):
    '''SQLLite adapter
    '''

    def __init__(self, existing_db_name=None, ent_to_idx=None, rel_to_idx=None):
        """
        Initialize the SQLiteAdapter class variables.

        :param existing_db_name: Existing database name to use. Provided that the db already has the persisted schema
        by the adapter the data is already mapped.
        :type existing_db_name: str
        :param ent_to_idx: Entity to idx mappings
        :type ent_to_idx: dict
        :param rel_to_idx: Relation to idx mappings
        :type rel_to_idx: dict
        """

        super(SQLiteAdapter, self).__init__()
        # persistence status of the data
        self.persistance_status = {}
        self.dbname = existing_db_name
        # flag indicating whether we are using existing db
        self.using_existing_db = False
        self.temp_dir = None
        if self.dbname is not None:
            # If we are using existing db then the mappings need to be passed
            assert (self.rel_to_idx is not None)
            assert (self.ent_to_idx is not None)

            self.using_existing_db = True
            self.rel_to_idx = rel_to_idx
            self.ent_to_idx = ent_to_idx