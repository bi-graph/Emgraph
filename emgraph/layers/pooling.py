import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sum_pooling(embeddings):
    """
    Sum pooling of all embeddings along neighbor axis.

    :param embeddings: Embedding of a list of subjects
    :type embeddings: Tensor, shape [B, max_rel, emb_dim]
    :return: Reduced vector v
    :rtype: tf.Operation
    """

    return tf.reduce_sum(embeddings, axis=1)


def avg_pooling(embeddings):
    """
    Avg pooling of all embeddings along neighbor axis.

    :param embeddings: Embedding of a list of subjects
    :type embeddings: Tensor, shape [B, max_rel, emb_dim]
    :return: Reduced vector v
    :rtype: tf.Operation
    """
    return tf.reduce_mean(embeddings, axis=1)


def max_pooling(embeddings):
    """
    Max pooling of all embeddings along neighbor axis.

    :param embeddings: Embedding of a list of subjects
    :type embeddings: Tensor, shape [B, max_rel, emb_dim]
    :return: Reduced vector v
    :rtype: tf.Operation
    """
    return tf.reduce_max(embeddings, axis=1)
