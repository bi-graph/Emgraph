from collections.abc import Iterable
from itertools import product, islice
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf


from ..evaluation import mrr_score, hits_at_n_score, mr_score
from ..datasets import BigraphDatasetAdapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOO_MANY_ENTITIES_TH = 50000


def _train_test_split_no_unseen_fast(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.

     This is an improved version which is much faster - since this doesnt sample like earlier approach but rather
     shuffles indices and gets the test set of required size by selecting from the shuffled indices only triples
     which do not disconnect entities/relations.

    :param X: The dataset to split.
    :type X: ndarray, size[n, 3]
    :param test_size: Number of triples in the test set (if it was int)
    The percentage of total triples (if it was float)
    :type test_size: int, float
    :param seed: 'Random' seed used in splitting the dataset
    :type seed: int
    :param allow_duplication: Flag to allow duplicates in the test set
    :type allow_duplication: bool
    :param filtered_test_predicates: If None, all predicate types will be considered for the test set.
        If it was a list, only the predicate types in the list will be considered in
        the test set.
    :type filtered_test_predicates: None, list
    :return: Training set, test set
    :rtype: ndarray, size[n, 3], ndarray, size[n, 3]

    Examples:

    >>> import numpy as np
    >>> from bigraph.evaluation import train_test_split_no_unseen
    >>> # load your dataset to X
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['f', 'y', 'e'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> # if you want to split into train/test datasets
    >>> X_train, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train
    array([['a', 'y', 'd'],
       ['b', 'y', 'a'],
       ['a', 'y', 'c'],
       ['f', 'y', 'e'],
       ['a', 'y', 'b'],
       ['c', 'y', 'a'],
       ['b', 'y', 'c']], dtype='<U1')
    >>> X_test
    array([['f', 'y', 'e'],
       ['c', 'y', 'd']], dtype='<U1')
    >>> # if you want to split into train/valid/test datasets, call it 2 times
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2)
    >>> X_train
    array([['a', 'y', 'b'],
       ['a', 'y', 'd'],
       ['a', 'y', 'c'],
       ['c', 'y', 'a'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_valid
    array([['c', 'y', 'd'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_test
    array([['b', 'y', 'c'],
       ['b', 'y', 'a']], dtype='<U1')
    """

    if type(test_size) is float:
        test_size = int(len(X) * test_size)

    np.random.seed(seed)
    if filtered_test_predicates:
        candidate_idx = np.isin(X[:, 1], filtered_test_predicates)
        X_test_candidates = X[candidate_idx]
        X_train = X[~candidate_idx]
    else:
        X_train = None
        X_test_candidates = X

    entities, entity_cnt = np.unique(np.concatenate([X_test_candidates[:, 0],
                                                     X_test_candidates[:, 2]]), return_counts=True)
    rels, rels_cnt = np.unique(X_test_candidates[:, 1], return_counts=True)
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []

    all_indices_shuffled = np.random.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        # reduce the entity and rel count
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0:

            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))

                # break out of the loop
                break

        else:
            # since removing this triple results in unseen entities, add it to training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            idx_train.append(idx)

    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = np.random.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True,"
                            "remove filter on test predicates or "
                            "set test_size to a smaller value.")

    if X_train is None:
        X_train = X_test_candidates[idx_train]
    else:
        X_train_subset = X_test_candidates[idx_train]
        X_train = np.concatenate([X_train, X_train_subset])
    X_test = X_test_candidates[idx_test]

    X_train = np.random.permutation(X_train)
    X_test = np.random.permutation(X_test)

    return X_train, X_test

def _create_unique_mappings(unique_obj, unique_rel):
    """Create unique mappings.

    :param unique_obj: Unique object
    :type unique_obj: list
    :param unique_rel: Unique relationship
    :type unique_rel: list
    :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping: entity to idx mapping
    :rtype: dict, dict
    """

    obj_count = len(unique_obj)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    obj_to_idx = dict(zip(unique_obj, range(obj_count)))
    return rel_to_idx, obj_to_idx


def create_mappings(X):
    """Create string-IDs mappings for entities and relations.

    Entities and relations are assigned incremental, unique integer IDs.
    Mappings are preserved in two distinct dictionaries,
    and counters are separated for entities and relations mappings.

    :param X: The triples to extract mappings
    :type X: ndarray, shape [n, 3]
    :return: Rel-to-idx: Relation to idx mapping - ent-to-idx mapping: entity to idx mapping
    :rtype: dict, dict
    """

    logger.debug('Creating mappings for entities and relations.')
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
    unique_rel = np.unique(X[:, 1])
    return _create_unique_mappings(unique_ent, unique_rel)

def _convert_to_idx(X, ent_to_idx, rel_to_idx, obj_to_idx):
    """Convert statements (triples) into integer IDs.

    :param X: The statements to be converted.
    :type X: ndarray
    :param ent_to_idx: Entity to idx mappings
    :type ent_to_idx: dict
    :param rel_to_idx: Relation to idx mappings
    :type rel_to_idx: dict
    :param obj_to_idx: Object to idx mappings
    :type obj_to_idx: dict
    :return: Converted statements
    :rtype: ndarray, shape [n, 3]
    """
    unseen_msg = 'Input triples include one or more {concept_type} not present in the training set. ' \
                 'Please filter all concepts in X that do not occur in the training test ' \
                 '(set filter_unseen=True in evaluate_performance) or retrain the model on a ' \
                 'training set that includes all the desired concept types.'

    try:
        x_idx_s = np.vectorize(ent_to_idx.get)(X[:, 0])
        x_idx_p = np.vectorize(rel_to_idx.get)(X[:, 1])
        x_idx_o = np.vectorize(obj_to_idx.get)(X[:, 2])
    except TypeError:
        unseen_msg = unseen_msg.format(**{'concept_type': 'concepts'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_s or None in x_idx_o:
        unseen_msg = unseen_msg.format(**{'concept_type': 'entities'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_p:
        unseen_msg = unseen_msg.format(**{'concept_type': 'relations'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    return np.dstack([x_idx_s, x_idx_p, x_idx_o]).reshape((-1, 3))


def to_idx(X, ent_to_idx, rel_to_idx):
    """Convert statements (triples) into integer IDs.

    :param X: The statements to be converted.
    :type X: ndarray
    :param ent_to_idx: Entity to idx mappings
    :type ent_to_idx: dict
    :param rel_to_idx: Relation to idx mappings
    :type rel_to_idx: dict
    :return: Converted statements
    :rtype: ndarray, shape [n, 3]
    """

    logger.debug('Converting statements to integer ids.')
    if X.ndim == 1:
        X = X[np.newaxis, :]
    return _convert_to_idx(X, ent_to_idx, rel_to_idx, ent_to_idx)