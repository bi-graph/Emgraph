
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

    def generate_mappings(self, use_all=False, regenerate=False):
        """
        Generate mappings from either train set or use all dataset to generate mappings.

        :param use_all: Whether to use all the data or not. If True it uses all the data else the train
        set (default: False)
        :type use_all: bool
        :param regenerate: Whether to regenerate the mappings from all the data. If True it regenerates the mappings.
        In that case it will recreate the database to add new mappings.
        :type regenerate: bool
        :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping
        :rtype: dict, dict
        """

        if (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0 or (regenerate is True)) \
                and (not self.using_existing_db):
            from ..evaluation import create_mappings
            self._create_schema()
            if use_all:
                complete_dataset = []
                for key in self.dataset.keys():
                    complete_dataset.append(self.dataset[key])
                self.rel_to_idx, self.ent_to_idx = create_mappings(np.concatenate(complete_dataset, axis=0))

            else:
                self.rel_to_idx, self.ent_to_idx = create_mappings(self.dataset["train"])

            self._insert_entities_in_db()
        return self.rel_to_idx, self.ent_to_idx

    def _insert_entities_in_db(self):
        """Insert entities in the database
        """
        # TODO: can change it to just use the values of the dictionary
        pg_entity_values = np.arange(len(self.ent_to_idx)).reshape(-1, 1).tolist()
        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        try:
            cur.executemany('INSERT INTO entity_table VALUES (?)', pg_entity_values)
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
        cur.close()
        conn.close()


    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Use an existing mapping with the datasource.
        """
        # cannot change mappings for an existing database.
        if self.using_existing_db:
            raise Exception('Cannot change the mappings for an existing DB')
        super().use_mappings(rel_to_idx, ent_to_idx)
        self._create_schema()

        for key in self.dataset.keys():
            self.mapped_status[key] = False
            self.persistance_status[key] = False

        self._insert_entities_in_db()

    def get_size(self, dataset_type="train"):
        """Return the size of the specified dataset.

        :param dataset_type: Dataset type
        :type dataset_type: str
        :return: Size of the specified dataset
        :rtype: int
        """

        select_query = "SELECT count(*) from triples_table where dataset_type ='{}'"
        conn = sqlite3.connect("{}".format(self.dbname))
        cur1 = conn.cursor()
        cur1.execute(select_query.format(dataset_type))
        out = cur1.fetchall()
        cur1.close()
        return out[0][0]

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

        if (not self.using_existing_db) and (not self.mapped_status[dataset_type]):
            self.map_data()

        if batches_count == -1:
            batch_size = 1
            batches_count = self.get_size(dataset_type)
        else:
            batch_size = int(np.ceil(self.get_size(dataset_type) / batches_count))

        select_query = "SELECT subject, predicate,object FROM triples_table INDEXED BY \
                            triples_table_type_idx where dataset_type ='{}' LIMIT {}, {}"

        for i in range(batches_count):
            conn = sqlite3.connect("{}".format(self.dbname))
            cur1 = conn.cursor()
            cur1.execute(select_query.format(dataset_type, i * batch_size, batch_size))
            out = np.array(cur1.fetchall(), dtype=np.int32)
            cur1.close()
            if use_filter:
                # get the filter values
                participating_objects, participating_subjects = self.get_participating_entities(out)
                yield out, participating_objects, participating_subjects
            else:
                yield out

    def _insert_triples(self, triples, key=""):
        """inserts triples in the database for the specified category

        :param triples: Triples of entity-relation-entity
        :type triples: set
        :param key: Key for the triple values
        :type key: str
        :return: -
        :rtype: -
        """

        conn = sqlite3.connect("{}".format(self.dbname))
        key = np.array([[key]])
        for j in range(int(np.ceil(triples.shape[0] / 500000.0))):
            pg_triple_values = triples[j * 500000:(j + 1) * 500000]
            pg_triple_values = np.concatenate((pg_triple_values, np.repeat(key,
                                                                           pg_triple_values.shape[0], axis=0)), axis=1)
            pg_triple_values = pg_triple_values.tolist()
            cur = conn.cursor()
            cur.executemany('INSERT INTO triples_table VALUES (?,?,?,?)', pg_triple_values)
            conn.commit()
            cur.close()

        conn.close()

    def map_data(self, remap=False):
        """Map the data to the mappings of ent_to_idx and rel_to_idx

        :param remap: Remap flag to update the mappings from the dictionary
        :type remap: bool
        :return: -
        :rtype: -
        """

        if self.using_existing_db:
            # since the assumption is that the persisted data is already mapped for an existing db
            return
        from ..evaluation import to_idx
        if len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0:
            self.generate_mappings()

        for key in self.dataset.keys():
            if isinstance(self.dataset[key], np.ndarray):
                if (not self.mapped_status[key]) or (remap is True):
                    self.dataset[key] = to_idx(self.dataset[key],
                                               ent_to_idx=self.ent_to_idx,
                                               rel_to_idx=self.rel_to_idx)
                    self.mapped_status[key] = True
                if not self.persistance_status[key]:
                    self._insert_triples(self.dataset[key], key)
                    self.persistance_status[key] = True

        conn = sqlite3.connect("{}".format(self.dbname))
        cur = conn.cursor()
        # to maintain integrity of data
        cur.execute('Update integrity_check set validity=1 where validity=0')
        conn.commit()

        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_ins_integrity_check_trigger
                        AFTER INSERT ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_upd_integrity_check_trigger
                        AFTER UPDATE ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS triples_table_del_integrity_check_trigger
                        AFTER DELETE ON triples_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                            ;
                    ''')

        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_upd_integrity_check_trigger
                        AFTER UPDATE ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_ins_integrity_check_trigger
                        AFTER INSERT ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS entity_table_del_integrity_check_trigger
                        AFTER DELETE ON entity_table
                        BEGIN
                            Update integrity_check set validity=0 where validity=1;
                        END
                        ;
                    ''')
        cur.close()
        conn.close()

