import numpy as np
import pytest
import os

from bigraph.latent_features import TransE, DistMult, ComplEx, HolE, RandomBaseline, ConvKB, ConvE
from bigraph.latent_features import set_entity_threshold, reset_entity_threshold
from bigraph.datasets import load_wn18, load_wn18rr
from bigraph.evaluation import evaluate_performance, hits_at_n_score
from bigraph.utils import save_model, restore_model
from bigraph.evaluation.protocol import to_idx


def test_conve_bce_combo():
    # no exception
    model = ConvE(loss='bce')

    # no exception
    model = TransE(loss='nll')

    # Invalid combination. Hence exception.
    with pytest.raises(ValueError):
        model = TransE(loss='bce')

    # Invalid combination. Hence exception.
    with pytest.raises(ValueError):
        model = ConvE(loss='nll')


def test_large_graph_mode():
    set_entity_threshold(10)
    X = load_wn18()
    model = ComplEx(batches_count=100, seed=555, epochs=1, k=50, loss='multiclass_nll', loss_params={'margin': 5},
                    verbose=True, optimizer='sgd', optimizer_params={'lr': 0.001})
    model.fit(X['train'])
    X_filter = np.concatenate((X['train'], X['valid'], X['test']), axis=0)
    evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o')

    y = model.predict(X['test'][:1])
    print(y)
    reset_entity_threshold()


def test_output_sizes():
    ''' Test to check whether embedding matrix sizes match the input data (num rel/ent and k)
    '''
    def perform_test():
        X = load_wn18rr()
        k = 5
        unique_entities = np.unique(np.concatenate([X['train'][:, 0],
                                                    X['train'][:, 2]], 0))
        unique_relations = np.unique(X['train'][:, 1])
        model = TransE(batches_count=100, seed=555, epochs=1, k=k, loss='multiclass_nll', loss_params={'margin': 5},
                        verbose=True, optimizer='sgd', optimizer_params={'lr': 0.001})
        model.fit(X['train'])
        # verify ent and rel shapes
        assert(model.trained_model_params[0].shape[0] == len(unique_entities))
        assert(model.trained_model_params[1].shape[0] == len(unique_relations))
        # verify k
        assert(model.trained_model_params[0].shape[1] == k)
        assert(model.trained_model_params[1].shape[1] == k)

    # Normal mode
    perform_test()

    # Large graph mode
    set_entity_threshold(10)
    perform_test()
    reset_entity_threshold()


def test_large_graph_mode_adam():
    set_entity_threshold(10)
    X = load_wn18()
    model = ComplEx(batches_count=100, seed=555, epochs=1, k=50, loss='multiclass_nll', loss_params={'margin': 5},
                    verbose=True, optimizer='adam', optimizer_params={'lr': 0.001})
    try:
        model.fit(X['train'])
    except Exception as e:
        print(str(e))

    reset_entity_threshold()


def test_fit_predict_TransE_early_stopping_with_filter():
    X = load_wn18()
    model = TransE(batches_count=1, seed=555, epochs=7, k=50, loss='pairwise', loss_params={'margin': 5},
                   verbose=True, optimizer='adagrad', optimizer_params={'lr': 0.1})
    X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    model.fit(X['train'], True, {'x_valid': X['valid'][::100],
                                 'criteria': 'mrr',
                                 'x_filter': X_filter,
                                 'stop_interval': 2,
                                 'burn_in': 1,
                                 'check_interval': 2})

    y = model.predict(X['test'][:1])
    print(y)


def test_fit_predict_TransE_early_stopping_without_filter():
    X = load_wn18()
    model = TransE(batches_count=1, seed=555, epochs=7, k=50, loss='pairwise', loss_params={'margin': 5},
                   verbose=True, optimizer='adagrad', optimizer_params={'lr': 0.1})
    model.fit(X['train'], True, {'x_valid': X['valid'][::100],
                                 'criteria': 'mrr',
                                 'stop_interval': 2,
                                 'burn_in': 1,
                                 'check_interval': 2})

    y = model.predict(X['test'][:1])
    print(y)


def test_evaluate_RandomBaseline():
    model = RandomBaseline(seed=0)
    X = load_wn18()
    model.fit(X["train"])
    ranks = evaluate_performance(X["test"],
                                 model=model,
                                 corrupt_side='s+o',
                                 verbose=False)
    hits10 = hits_at_n_score(ranks, n=10)
    hits1 = hits_at_n_score(ranks, n=1)
    assert ranks.shape == (len(X['test']), )
    assert hits10 < 0.01 and hits1 == 0.0

    ranks = evaluate_performance(X["test"],
                                 model=model,
                                 corrupt_side='s,o',
                                 verbose=False)
    hits10 = hits_at_n_score(ranks, n=10)
    hits1 = hits_at_n_score(ranks, n=1)
    assert ranks.shape == (len(X['test']), 2)
    assert hits10 < 0.01 and hits1 == 0.0

    ranks_filtered = evaluate_performance(X["test"],
                                          filter_triples=np.concatenate((X['train'], X['valid'], X['test'])),
                                          model=model,
                                          corrupt_side='s,o',
                                          verbose=False)
    hits10 = hits_at_n_score(ranks_filtered, n=10)
    hits1 = hits_at_n_score(ranks_filtered, n=1)
    assert ranks_filtered.shape == (len(X['test']), 2)
    assert hits10 < 0.01 and hits1 == 0.0
    assert np.all(ranks_filtered <= ranks)
    assert np.any(ranks_filtered != ranks)


def test_fit_predict_transE():
    model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin': 5},
                   optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]


def test_fit_predict_DistMult():
    model = DistMult(batches_count=2, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]

