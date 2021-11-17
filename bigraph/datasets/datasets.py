import hashlib
import os
import urllib
import zipfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BIGRAPH_ENV_NAME = 'BIGRAPH_DATA_HOME'
DatasetMetadata = namedtuple('DatasetMetadata', ['dataset_name', 'filename', 'url', 'train_name', 'valid_name',
                                                 'test_name', 'train_checksum', 'valid_checksum', 'test_checksum'])


def _clean_data(X, return_idx=False):

    if X["train"].shape[1] == 3:
        columns = ['s', 'p', 'o']
    else:
        columns = ['s', 'p', 'o', 'w']

    train = pd.DataFrame(X["train"], columns=columns)
    valid = pd.DataFrame(X["valid"], columns=columns)
    test = pd.DataFrame(X["test"], columns=columns)

    train_ent = np.unique(np.concatenate((train.s, train.o)))
    train_rel = train.p.unique()

    valid_idx = valid.s.isin(train_ent) & valid.o.isin(train_ent) & valid.p.isin(train_rel)
    test_idx = test.s.isin(train_ent) & test.o.isin(train_ent) & test.p.isin(train_rel)

    filtered_valid = valid[valid_idx].values
    filtered_test = test[test_idx].values

    filtered_X = {'train': train.values, 'valid': filtered_valid, 'test': filtered_test}

    if return_idx:
        return filtered_X, valid_idx, test_idx
    else:
        return filtered_X

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

def _load_dataset(dataset_metadata, data_home=None, check_md5hash=False, add_reciprocal_rels=False):

    if dataset_metadata.dataset_name is None:
        if dataset_metadata.url is None:
            raise ValueError('The dataset name or url must be provided to load a dataset.')
        dataset_metadata.dataset_name = dataset_metadata.url[dataset_metadata.url.rfind('/') + 1:dataset_metadata
                                                             .url.rfind('.')]
    dataset_path = _fetch_dataset(dataset_metadata, data_home, check_md5hash)

    train = load_from_csv(dataset_path,
                          dataset_metadata.train_name,
                          add_reciprocal_rels=add_reciprocal_rels)
    valid = load_from_csv(dataset_path,
                          dataset_metadata.valid_name,
                          add_reciprocal_rels=add_reciprocal_rels)
    test = load_from_csv(dataset_path,
                         dataset_metadata.test_name,
                         add_reciprocal_rels=add_reciprocal_rels)

    return {'train': train, 'valid': valid, 'test': test}


def load_wn18(check_md5hash=False, add_reciprocal_rels=False):


    wn18 = DatasetMetadata(
        dataset_name='wn18',
        filename='wn18.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wn18.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='7d68324d293837ac165c3441a6c8b0eb',
        valid_checksum='f4f66fec0ca83b5ebe7ad7003404e61d',
        test_checksum='b035247a8916c7ec3443fa949e1ff02c'
    )

    return _load_dataset(wn18,
                         data_home=None,
                         check_md5hash=check_md5hash,
                         add_reciprocal_rels=add_reciprocal_rels)


def load_wn18rr(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):


    wn18rr = DatasetMetadata(
        dataset_name='wn18RR',
        filename='wn18RR.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wn18RR.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='35e81af3ae233327c52a87f23b30ad3c',
        valid_checksum='74a2ee9eca9a8d31f1a7d4d95b5e0887',
        test_checksum='2b45ba1ba436b9d4ff27f1d3511224c9'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(wn18rr,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(wn18rr,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)


def load_fb15k(check_md5hash=False, add_reciprocal_rels=False):

    FB15K = DatasetMetadata(
        dataset_name='fb15k',
        filename='fb15k.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/fb15k.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='5a87195e68d7797af00e137a7f6929f2',
        valid_checksum='275835062bb86a86477a3c402d20b814',
        test_checksum='71098693b0efcfb8ac6cd61cf3a3b505'
    )

    return _load_dataset(FB15K,
                         data_home=None,
                         check_md5hash=check_md5hash,
                         add_reciprocal_rels=add_reciprocal_rels)


def load_fb15k_237(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):

    fb15k_237 = DatasetMetadata(
        dataset_name='fb15k-237',
        filename='fb15k-237.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/fb15k-237.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='c05b87b9ac00f41901e016a2092d7837',
        valid_checksum='6a94efd530e5f43fcf84f50bc6d37b69',
        test_checksum='f5bdf63db39f455dec0ed259bb6f8628'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(fb15k_237,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(fb15k_237,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)

def load_yago3_10(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):

    yago3_10 = DatasetMetadata(
        dataset_name='YAGO3-10',
        filename='YAGO3-10.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/YAGO3-10.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='a9da8f583ec3920570eeccf07199229a',
        valid_checksum='2d679a906f2b1ac29d74d5c948c1ad09',
        test_checksum='14bf97890b2fee774dbce5f326acd189'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(yago3_10,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(yago3_10,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)


def load_wn11(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):

    wn11 = DatasetMetadata(
        dataset_name='wordnet11',
        filename='wordnet11.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wordnet11.zip',
        train_name='train.txt',
        valid_name='dev.txt',
        test_name='test.txt',
        train_checksum='2429c672c89e33ad4fa8e1a3ade416e4',
        valid_checksum='87bf86e225e79294a2524089614b96aa',
        test_checksum='24113b464f8042c339e3e6833c1cebdf'
    )

    dataset = _load_dataset(wn11, data_home=None,
                            check_md5hash=check_md5hash,
                            add_reciprocal_rels=add_reciprocal_rels)

    valid_labels = dataset['valid'][:, 3]
    test_labels = dataset['test'][:, 3]

    dataset['valid'] = dataset['valid'][:, 0:3]
    dataset['test'] = dataset['test'][:, 0:3]

    dataset['valid_labels'] = valid_labels == '1'
    dataset['test_labels'] = test_labels == '1'

    if clean_unseen:
        clean_dataset, valid_idx, test_idx = _clean_data(dataset, return_idx=True)
        clean_dataset['valid_labels'] = dataset['valid_labels'][valid_idx]
        clean_dataset['test_labels'] = dataset['test_labels'][test_idx]
        return clean_dataset
    else:
        return dataset


def load_fb13(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):

    fb13 = DatasetMetadata(
        dataset_name='freebase13',
        filename='freebase13.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/freebase13.zip',
        train_name='train.txt',
        valid_name='dev.txt',
        test_name='test.txt',
        train_checksum='9099ebcd85ab3ce723cfaaf34f74dceb',
        valid_checksum='c4ef7b244baa436a97c2a5e57d4ba7ed',
        test_checksum='f9af2eac7c5a86996c909bdffd295528'
    )

    dataset = _load_dataset(fb13,
                            data_home=None,
                            check_md5hash=check_md5hash,
                            add_reciprocal_rels=add_reciprocal_rels)

    valid_labels = dataset['valid'][:, 3]
    test_labels = dataset['test'][:, 3]

    dataset['valid'] = dataset['valid'][:, 0:3]
    dataset['test'] = dataset['test'][:, 0:3]

    dataset['valid_labels'] = valid_labels == '1'
    dataset['test_labels'] = test_labels == '1'

    if clean_unseen:
        clean_dataset, valid_idx, test_idx = _clean_data(dataset, return_idx=True)
        clean_dataset['valid_labels'] = dataset['valid_labels'][valid_idx]
        clean_dataset['test_labels'] = dataset['test_labels'][test_idx]
        return clean_dataset
    else:
        return dataset

def load_all_datasets(check_md5hash=False):
    load_wn18(check_md5hash)
    load_wn18rr(check_md5hash)
    load_fb15k(check_md5hash)
    load_fb15k_237(check_md5hash)
    load_yago3_10(check_md5hash)
    load_wn11(check_md5hash)
    load_fb13(check_md5hash)
