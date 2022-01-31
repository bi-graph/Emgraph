import numpy as np
import logging

"""This module contains learning-to-rank metrics to evaluate the performance of neural graph embedding models."""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def hits_at_n_score(ranks, n):
    """Number of ranked vector ``ranks``'s elements in the top ``n`` positions.

    It can be used in conjunction with the learning to rank evaluation protocol of
    :meth:`bigraph.evaluation.evaluate_performance`.

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

    Examples
    --------
    >>> import numpy as np
    >>> from bigraph.evaluation.metrics import hits_at_n_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> hits_at_n_score(rankings, n=3)
    0.5
    """

    logger.debug('Calculating hits-at-n.')
    if isinstance(ranks, list):
        logger.debug('Converting ranks to numpy array.')
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(ranks <= n) / len(ranks)
