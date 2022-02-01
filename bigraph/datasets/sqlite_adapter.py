
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

    def get_db_name(self):
        """Returns the db name
        """
        return self.dbname

    def _create_schema(self):
        """Create the database schema
        """
        if self.using_existing_db:
            return
        if self.dbname is not None:
            self.cleanup()

        self.temp_dir = tempfile.TemporaryDirectory(suffix=None, prefix='ampligraph_', dir=None)
        self.dbname = os.path.join(self.temp_dir.name, 'Ampligraph_{}.db'.format(int(time.time())))

        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        cur.execute("CREATE TABLE entity_table (entity_type integer primary key);")
        cur.execute("CREATE TABLE triples_table (subject integer, \
                                                    predicate integer, \
                                                    object integer, \
                                                    dataset_type text(50), \
                                                    foreign key (object) references entity_table(entity_type), \
                                                    foreign key (subject) references entity_table(entity_type) \
                                                    );")

        cur.execute("CREATE INDEX triples_table_sp_idx ON triples_table (subject, predicate);")
        cur.execute("CREATE INDEX triples_table_po_idx ON triples_table (predicate, object);")
        cur.execute("CREATE INDEX triples_table_type_idx ON triples_table (dataset_type);")

        cur.execute("CREATE TABLE integrity_check (validity integer primary key);")

        cur.execute('INSERT INTO integrity_check VALUES (0)')
        conn.commit()
        cur.close()
        conn.close()