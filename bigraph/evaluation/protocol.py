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


def _train_test_split_no_unseen_old(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.

     This is very slow as it runs an infinite loop and samples a triples and appends to test set and checks if it is
     unique or not. This is very time consuming process and highly inefficient.

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
    >>> X_train, X_test = train_test_split_no_unseen(X, test_size=2, backward_compatible=True)
    >>> X_train
    array([['a', 'y', 'b'],
        ['f', 'y', 'e'],
        ['b', 'y', 'a'],
        ['c', 'y', 'a'],
        ['c', 'y', 'd'],
        ['b', 'y', 'c'],
        ['f', 'y', 'e']], dtype='<U1')
    >>> X_test
    array([['a', 'y', 'c'],
        ['a', 'y', 'd']], dtype='<U1')
    >>> # if you want to split into train/valid/test datasets, call it 2 times
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2, backward_compatible=True)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2, backward_compatible=True)
    >>> X_train
    array([['a', 'y', 'b'],
        ['b', 'y', 'a'],
        ['c', 'y', 'd'],
        ['b', 'y', 'c'],
        ['f', 'y', 'e']], dtype='<U1')
    >>> X_valid
    array([['f', 'y', 'e'],
        ['c', 'y', 'a']], dtype='<U1')
    >>> X_test
    array([['a', 'y', 'c'],
        ['a', 'y', 'd']], dtype='<U1')
    """
    """
    
    """

    logger.debug('Creating train test split.')
    if type(test_size) is float:
        logger.debug('Test size is of type float. Converting to int.')
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    subs, subs_cnt = np.unique(X[:, 0], return_counts=True)
    objs, objs_cnt = np.unique(X[:, 2], return_counts=True)
    rels, rels_cnt = np.unique(X[:, 1], return_counts=True)
    dict_subs = dict(zip(subs, subs_cnt))
    dict_objs = dict(zip(objs, objs_cnt))
    dict_rels = dict(zip(rels, rels_cnt))

    idx_test = np.array([], dtype=int)
    logger.debug('Selecting test cases using random search.')

    loop_count = 0
    tolerance = len(X) * 10
    # Set the indices of test set triples. If filtered, reduce candidate triples to certain predicate types.
    if filtered_test_predicates:
        test_triples_idx = np.where(np.isin(X[:, 1], filtered_test_predicates))[0]
    else:
        test_triples_idx = np.arange(len(X))

    while idx_test.shape[0] < test_size:
        i = rnd.choice(test_triples_idx)
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i, 1]] > 1:
            dict_subs[X[i, 0]] -= 1
            dict_objs[X[i, 2]] -= 1
            dict_rels[X[i, 1]] -= 1
            if allow_duplication:
                idx_test = np.append(idx_test, i)
            else:
                idx_test = np.unique(np.append(idx_test, i))

        loop_count += 1

        # in case can't find solution
        if loop_count == tolerance:
            if allow_duplication:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Change seed values, remove filter on test predicates or set "
                                "test_size to a smaller value.")
            else:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Set allow_duplication=True,"
                                "change seed values, remove filter on test predicates or "
                                "set test_size to a smaller value.")

    logger.debug('Completed random search.')

    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)
    logger.debug('Train test split completed.')

    return X[idx_train, :], X[idx_test, :]


def train_test_split_no_unseen(X, test_size=100, seed=0, allow_duplication=False,
                               filtered_test_predicates=None, backward_compatible=False):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.


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
    :param backward_compatible: Uses the old (slower) version of the API for reproducibility of splits in older pipelines(if any)
        Avoid setting this to True, unless necessary. Set this flag only if you want to use the
        train_test_split_no_unseen of Ampligraph versions 1.3.2 and below. The older version is slow and inefficient
    :type backward_compatible: bool
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

    if backward_compatible:
        return _train_test_split_no_unseen_old(X, test_size, seed, allow_duplication, filtered_test_predicates)

    return _train_test_split_no_unseen_fast(X, test_size, seed, allow_duplication, filtered_test_predicates)


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


def generate_corruptions_for_eval(X, entities_for_corruption, corrupt_side='s,o'):
    """

    :param X: Currently, a single positive triples that will be used to create corruptions
    :type X: Tensor, shape [1, 3]
    :param entities_for_corruption: All the entity IDs which are to be used for generation of corruptions
    :type entities_for_corruption: tensor
    :param corrupt_side: Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
        - 's,o': corrupt both subject and object but ranks are computed separately.
    :type corrupt_side: str
    :return: An array of corruptions for the triples for x.
    :rtype: Tensor, shape [1, 3]
    """

    logger.debug('Generating corruptions for evaluation.')

    logger.debug('Getting repeating subjects.')
    if corrupt_side == 's,o':
        # Both subject and object are corrupted but ranks are computed separately.
        corrupt_side = 's+o'

    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value for corruption side passed for evaluation'
        logger.error(msg)
        raise ValueError(msg)

    if corrupt_side in ['s+o', 'o']:  # object is corrupted - so we need subjects as it is
        repeated_subjs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 0],  # subj
                     [tf.shape(X)[0], 1]),
            tf.shape(entities_for_corruption)[0])
        repeated_subjs = tf.squeeze(repeated_subjs, 2)

    logger.debug('Getting repeating object.')
    if corrupt_side in ['s+o', 's']:  # subject is corrupted - so we need objects as it is
        repeated_objs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 2],  # Obj
                     [tf.shape(X)[0], 1]),
            tf.shape(entities_for_corruption)[0])
        repeated_objs = tf.squeeze(repeated_objs, 2)

    logger.debug('Getting repeating relationships.')
    repeated_relns = tf.keras.backend.repeat(
        tf.slice(X,
                 [0, 1],  # reln
                 [tf.shape(X)[0], 1]),
        tf.shape(entities_for_corruption)[0])
    repeated_relns = tf.squeeze(repeated_relns, 2)

    rep_ent = tf.keras.backend.repeat(tf.expand_dims(entities_for_corruption, 0), tf.shape(X)[0])
    rep_ent = tf.squeeze(rep_ent, 0)

    if corrupt_side == 's+o':
        stacked_out = tf.concat([tf.stack([repeated_subjs, repeated_relns, rep_ent], 1),
                                 tf.stack([rep_ent, repeated_relns, repeated_objs], 1)], 0)

    elif corrupt_side == 'o':
        stacked_out = tf.stack([repeated_subjs, repeated_relns, rep_ent], 1)

    else:
        stacked_out = tf.stack([rep_ent, repeated_relns, repeated_objs], 1)

    out = tf.reshape(tf.transpose(stacked_out, [0, 2, 1]), (-1, 3))

    return out


def generate_corruptions_for_fit(X, entities_list=None, eta=1, corrupt_side='s,o', entities_size=0, rnd=None):
    """Generate corruptions for training.

    Creates corrupted triples for each statement in an array of statements,
    as described by :cite:`trouillon2016complex`.

    .. note::
        Collisions are not checked, as this will be computationally expensive :cite:`trouillon2016complex`.
        That means that some corruptions *may* result in being positive statements (i.e. *unfiltered* settings).

    .. note::
        When processing large knowledge graphs, it may be useful to generate corruptions only using entities from
        a single batch.
        This also brings the benefit of creating more meaningful negatives, as entities used to corrupt are
        sourced locally.
        The function can be configured to generate corruptions *only* using the entities from the current batch.
        You can enable such behaviour be setting ``entities_size=0``. In such case, if ``entities_list=None``
        all entities from the *current batch* will be used to generate corruptions.

    :param X: An array of positive triples that will be used to create corruptions.
    :type X: Tensor, shape [n, 3]
    :param entities_list: List of entities to be used for generating corruptions. (default:None).
        If ``entities_list=None`` and ``entities_size`` is the number of all entities,
        all entities will be used to generate corruptions (default behaviour).

        If ``entities_list=None`` and ``entities_size=0``, the batch entities will be used to generate corruptions.
    :type entities_list: list
    :param eta: Number of corruptions per triple that must be generated
    :type eta: int
    :param corrupt_side: Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
        - 's,o': corrupt both subject and object
    :type corrupt_side: str
    :param entities_size: Size of entities to be used while generating corruptions. It assumes entity id's start from 0 and are
        continuous. (default: 0).
        When processing large knowledge graphs, it may be useful to generate corruptions only using entities from
        a single batch.
        This also brings the benefit of creating more meaningful negatives, as entities used to corrupt are
        sourced locally.
        The function can be configured to generate corruptions *only* using the entities from the current batch.
        You can enable such behaviour be setting ``entities_size=0``. In such case, if ``entities_list=None``
        all entities from the *current batch* will be used to generate corruptions.
    :type entities_size: int
    :param rnd: A random number generator.
    :type rnd: numpy.random.RandomState
    :return: An array of corruptions for a list of positive triples X. For each row in X the corresponding corruption
        indexes can be found at [index+i*n for i in range(eta)]

    :rtype: tensor, shape [n * eta, 3]
    """

    logger.debug('Generating corruptions for fit.')
    if corrupt_side == 's,o':
        # Both subject and object are corrupted but ranks are computed separately.
        corrupt_side = 's+o'

    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value {} for corruption side passed for evaluation.'.format(corrupt_side)
        logger.error(msg)
        raise ValueError(msg)

    dataset = tf.reshape(tf.tile(tf.reshape(X, [-1]), [eta]), [tf.shape(X)[0] * eta, 3])

    if corrupt_side == 's+o':
        keep_subj_mask = tf.cast(tf.random_uniform([tf.shape(X)[0] * eta], 0, 2, dtype=tf.int32, seed=rnd), tf.bool)
    else:
        keep_subj_mask = tf.cast(tf.ones(tf.shape(X)[0] * eta, tf.int32), tf.bool)
        if corrupt_side == 's':
            keep_subj_mask = tf.logical_not(keep_subj_mask)

    keep_obj_mask = tf.logical_not(keep_subj_mask)
    keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
    keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)

    logger.debug('Created corruption masks.')

    if entities_size != 0:
        replacements = tf.random_uniform([tf.shape(dataset)[0]], 0, entities_size, dtype=tf.int32, seed=rnd)
    else:
        if entities_list is None:
            # use entities in the batch
            entities_list, _ = tf.unique(tf.squeeze(
                tf.concat([tf.slice(X, [0, 0], [tf.shape(X)[0], 1]),
                           tf.slice(X, [0, 2], [tf.shape(X)[0], 1])],
                          0)))

        random_indices = tf.random.uniform(shape=(tf.shape(dataset)[0],),
                                           maxval=tf.shape(entities_list)[0],
                                           dtype=tf.int32,
                                           seed=rnd)
        replacements = tf.gather(entities_list, random_indices)

    subjects = tf.math.add(tf.math.multiply(keep_subj_mask, dataset[:, 0]),
                           tf.math.multiply(keep_obj_mask, replacements))
    logger.debug('Created corrupted subjects.')
    relationships = dataset[:, 1]
    logger.debug('Retained relationships.')
    objects = tf.math.add(tf.math.multiply(keep_obj_mask, dataset[:, 2]),
                          tf.math.multiply(keep_subj_mask, replacements))
    logger.debug('Created corrupted objects.')

    out = tf.transpose(tf.stack([subjects, relationships, objects]))

    logger.debug('Returning corruptions for fit.')
    return out








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