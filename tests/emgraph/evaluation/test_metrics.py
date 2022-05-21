import numpy as np

from emgraph.evaluation.metrics import hits_at_n_score, mr_score, mrr_score, rank_score


def test_rank_score():
    y_pred = np.array([0.434, 0.65, 0.21, 0.84])
    y_true = np.array([0, 0, 1, 0])
    rank_actual = rank_score(y_true, y_pred)
    assert rank_actual == 4


def test_mrr_score():
    y_pred_true = np.array(
        [[[0, 1, 0], [0.32, 0.84, 0.73]], [[0, 1, 0], [0.66, 0.11, 0.33]]]
    )

    rankings = []
    for y_pred_true_k in y_pred_true:
        rankings.append(rank_score(y_pred_true_k[0], y_pred_true_k[1]))
    mrr_actual = mrr_score(rankings)
    np.testing.assert_almost_equal(mrr_actual, 0.66666, decimal=5)


def test_hits_at_n_score():
    y_pred_true = np.array(
        [[[0, 1, 0], [0.32, 0.84, 0.73]], [[0, 1, 0], [0.66, 0.11, 0.33]]]
    )
    rankings = []
    for y_pred_true_k in y_pred_true:
        rankings.append(rank_score(y_pred_true_k[0], y_pred_true_k[1]))
    hits_actual = hits_at_n_score(rankings, n=2)
    assert hits_actual == 0.5


def test_mr_score():
    rank = np.array([0.2, 0.4, 0.6, 0.8])
    mr = mr_score(rank)
    assert mr == 0.5
