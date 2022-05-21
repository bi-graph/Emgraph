import logging

import numpy as np

"""This module contains learning-to-rank metrics to evaluate the performance of neural graph embedding models."""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def hits_at_n_score(ranks, n):
    """Number of ranked vector ``ranks``'s elements in the top ``n`` positions.

    It can be used in conjunction with the learning to rank evaluation protocol of
    :meth:`emgraph.evaluation.evaluate_performance`.

    It is formally defined as follows:

        .. math::

            HitsAtN = \sum_{i = 1}^{|Q|} 1 \, \text{if } rank_{(s, p, o)_i} \leq N

    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.


    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. HitsAt1 and HitsAt3 are:

        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5

        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5

        HitsAt3=1.0
        HitsAt1=0.5

    :param ranks: Ranking of ``n`` test statements
    :type ranks: list or ndarray, shape [n] or [n, 2]
    :param n: Maximum rank considered as positive
    :type n: int
    :return: The hits-at-n score
    :rtype: float

    Examples:
    >>> import numpy as np
    >>> from emgraph.evaluation.metrics import hits_at_n_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> hits_at_n_score(rankings, n=3)
    0.5
    """

    logger.debug("Calculating hits-at-n.")
    if isinstance(ranks, list):
        logger.debug("Converting ranks to numpy array.")
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(ranks <= n) / len(ranks)


def mrr_score(ranks):
    r"""Mean Reciprocal Rank (MRR)

    The function computes the mean of the reciprocal of elements of a vector of rankings ``ranks``.

    It is used in conjunction with the learning to rank evaluation protocol of
    :meth:`emgraph.evaluation.evaluate_performance`.

    It is formally defined as follows:

    .. math::

        MRR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}\frac{1}{rank_{(s, p, o)_i}}

    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.

    .. note::
        This metric is similar to mean rank (MR) :meth:`emgraph.evaluation.mr_score`. Instead of averaging ranks,
        it averages their reciprocals. This is done to obtain a metric which is more robust to outliers.


    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. The resulting MRR is: ::

        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5

        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5

        MRR=0.75

    :param ranks: Ranking of ``n`` test statements
    :type ranks: list or ndarray, shape [n] or [n, 2]
    :return: The MRR score
    :rtype: float

    Examples:
    >>> import numpy as np
    >>> from emgraph.evaluation.metrics import mrr_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> mrr_score(rankings)
    0.4375

    """

    logger.debug("Calculating the Mean Reciprocal Rank.")
    if isinstance(ranks, list):
        logger.debug("Converting ranks to numpy array.")
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(1 / ranks) / len(ranks)


def rank_score(y_true, y_pred, pos_lab=1):
    r"""Rank of a triple

        The rank of a positive element against a list of negatives.

    .. math::

        rank_{(s, p, o)_i}

    :param y_true: An array of binary labels each containing only one positive value.
    :type y_true: ndarray, shape[n]
    :param y_pred: An array of scores for the positive value and the n-1 negatives
    :type y_pred: ndaray, shape[n]
    :param pos_lab: The value of the positive label (default = 1)
    :type pos_lab: int
    :return: The rank of the positive element against the negatives
    :rtype: int

    Examples:
    >>> import numpy as np
    >>> from emgraph.evaluation.metrics import rank_score
    >>> y_pred = np.array([.434, .65, .21, .84])
    >>> y_true = np.array([0, 0, 1, 0])
    >>> rank_score(y_true, y_pred)
    4
    """

    logger.debug("Calculating the Rank Score.")
    idx = np.argsort(y_pred)[::-1]
    y_ord = y_true[idx]
    rank = np.where(y_ord == pos_lab)[0][0] + 1
    return rank


def mr_score(ranks):
    r"""Mean Rank (MR)

    The function computes the mean of of a vector of rankings ``ranks``.

    It can be used in conjunction with the learning to rank evaluation protocol of
    :meth:`emgraph.evaluation.evaluate_performance`.

    It is formally defined as follows:

    .. math::
        MR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}rank_{(s, p, o)_i}

    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.

    .. note::
        This metric is not robust to outliers.
        It is usually presented along the more reliable MRR :meth:`emgraph.evaluation.mrr_score`.

    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. The resulting MR is: ::

        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5

        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5

        MR=1.5

    :param ranks: Ranking of ``n`` test statements
    :type ranks: list or ndarray, shape [n] or [n, 2]
    :return: The MR score
    :rtype: float

    Examples:
    >>> from emgraph.evaluation import mr_score
    >>> ranks= [5, 3, 4, 10, 1]
    >>> mr_score(ranks)
    4.6
    """

    logger.debug("Calculating the Mean Average Rank score.")
    if isinstance(ranks, list):
        logger.debug("Converting ranks to numpy array.")
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(ranks) / len(ranks)
