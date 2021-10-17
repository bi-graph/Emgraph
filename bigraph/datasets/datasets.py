import hashlib
import os
import urllib
import zipfile
from pathlib import Path

import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BIGRAPH_ENV_NAME = 'AMPLIGRAPH_DATA_HOME'

def _get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get(BIGRAPH_ENV_NAME, os.path.join(os.getcwd(), 'bigraph_datasets'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    logger.debug('data_home is set to {}'.format(data_home))
    return data_home

def _md5(file_path):
    md5hash = hashlib.md5()
    chunk_size = 4096
    with open(file_path, 'rb') as f:
        content_buffer = f.read(chunk_size)
        while content_buffer:
            md5hash.update(content_buffer)
            content_buffer = f.read(chunk_size)
    return md5hash.hexdigest()

def _unzip_dataset(remote, source, destination, check_md5hash=False):
    # TODO - add error checking
    with zipfile.ZipFile(source, 'r') as zip_ref:
        logger.debug('Unzipping {} to {}'.format(source, destination))
        zip_ref.extractall(destination)
    if check_md5hash:
        for file_name, remote_checksum in [[remote.train_name, remote.train_checksum],
                                           [remote.valid_name, remote.valid_checksum],
                                           [remote.test_name, remote.test_checksum]]:
            file_path = os.path.join(destination, remote.dataset_name, file_name)
            checksum = _md5(file_path)
            if checksum != remote_checksum:
                os.remove(source)
                msg = '{} has an md5 checksum of ({}) which is different from the expected ({}), ' \
                      'the file may be corrupted.'.format(file_path, checksum, remote_checksum)
                logger.error(msg)
                raise IOError(msg)
    os.remove(source)

def _fetch_remote_data(remote, download_dir, data_home, check_md5hash=False):

    file_path = '{}.zip'.format(download_dir)
    if not Path(file_path).exists():
        urllib.request.urlretrieve(remote.url, file_path)
        # TODO - add error checking
    _unzip_dataset(remote, file_path, data_home, check_md5hash)

def _fetch_dataset(remote, data_home=None, check_md5hash=False):
    data_home = _get_data_home(data_home)
    dataset_dir = os.path.join(data_home, remote.dataset_name)
    if not os.path.exists(dataset_dir):
        if remote.url is None:
            msg = 'No dataset at {} and no url provided.'.format(dataset_dir)
            logger.error(msg)
            raise Exception(msg)

        _fetch_remote_data(remote, dataset_dir, data_home, check_md5hash)
    return dataset_dir

def _add_reciprocal_relations(tri_df):
    """
    Add reciprocal relations to the triples.

    :param tri_df: Dataframe of triples
    :type tri_df: Dataframe
    :return: Dataframe of triples and their reciprocals
    :rtype: Dataframe
    """
    df_reciprocal = tri_df.copy()
    # swap subjects and objects
    cols = list(df_reciprocal.columns)
    cols[0], cols[2] = cols[2], cols[0]
    df_reciprocal.columns = cols
    # add reciprocal relations
    df_reciprocal.iloc[:, 1] = df_reciprocal.iloc[:, 1] + "_reciprocal"
    # append to original triples
    tri_df = tri_df.append(df_reciprocal)
    return tri_df


def load_from_csv(directory_path, file_name, sep='\t', header=None, add_reciprocal_rels=False):
    """
    Load data from a CSV file as:
    .. code-block:: text

       subj1    relation1   obj1
       subj1    relation2   obj2
       subj3    relation3   obj2
       subj4    relation1   obj2
    ...

    :param directory_path: Input directory path
    :type directory_path: str
    :param file_name: File name
    :type file_name: str
    :param sep: Seperator to split columns by (default \t)
    :type sep: str
    :param header: CSV file header row (like in pandas)
    :type header: int, None
    :param add_reciprocal_rels: Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the
    dataset this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False)
    :type add_reciprocal_rels: bool
    :return: The actual triples of the file
    :rtype: ndarray , shape [n, 3]
    """

    logger.debug('Loading data from {}.'.format(file_name))
    df = pd.read_csv(os.path.join(directory_path, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    if add_reciprocal_rels:
        df = _add_reciprocal_relations(df)

    return df.values

