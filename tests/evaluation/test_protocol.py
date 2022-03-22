import numpy as np
import pytest
from itertools import islice

from bigraph.latent_features import TransE, ComplEx, RandomBaseline, set_entity_threshold, reset_entity_threshold
from bigraph.evaluation import evaluate_performance, generate_corruptions_for_eval, \
    generate_corruptions_for_fit, to_idx, create_mappings, mrr_score, hits_at_n_score, select_best_model_ranking, \
    filter_unseen_entities

from bigraph.datasets import load_wn18, load_wn18rr, load_yago3_10, load_fb15k_237
import tensorflow as tf

from bigraph.evaluation import train_test_split_no_unseen
from bigraph.evaluation.protocol import _next_hyperparam, _next_hyperparam_random, _remove_unused_params, \
    ParamHistory, _get_param_hash, _sample_parameters, _scalars_into_lists, _flatten_nested_keys, _unflatten_nested_keys


# test for #186
def test_evaluate_performance_too_many_entities_warning():
    X = load_yago3_10()
    model = TransE(batches_count=200, seed=0, epochs=1, k=5, eta=1, verbose=True)
    model.fit(X['train'])

    # no entity list declared
    with pytest.warns(UserWarning):
        evaluate_performance(X['test'][::100], model, verbose=True, corrupt_side='o')

    # with larger than threshold entity list
    with pytest.warns(UserWarning):
        # TOO_MANY_ENT_TH threshold is set to 50,000 entities. Using explicit value to comply with linting
        # and thus avoiding exporting unused global variable.
        entities_subset = np.union1d(np.unique(X["train"][:, 0]), np.unique(X["train"][:, 2]))[:50000]
        evaluate_performance(X['test'][::100], model, verbose=True, corrupt_side='o', entities_subset=entities_subset)

    # with small entity list (no exception expected)
    evaluate_performance(X['test'][::100], model, verbose=True, corrupt_side='o', entities_subset=entities_subset[:10])

    # with smaller dataset, no entity list declared (no exception expected)
    X_wn18rr = load_wn18rr()
    model_wn18 = TransE(batches_count=200, seed=0, epochs=1, k=5, eta=1, verbose=True)
    model_wn18.fit(X_wn18rr['train'])
    evaluate_performance(X_wn18rr['test'][::100], model_wn18, verbose=True, corrupt_side='o')


def test_evaluate_performance_filter_without_xtest():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=1, k=20, eta=10, loss='nll',
                    regularizer=None, optimizer='adam', optimizer_params={'lr': 0.01}, verbose=True)
    model.fit(X['train'])

    X_filter = np.concatenate((X['train'], X['valid']))  # filter does not contain X_test
    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o')
    assert (mrr_score(ranks) > 0)


def test_evaluate_performance_ranking_against_specified_entities():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=1, k=20, eta=10, loss='nll',
                    regularizer=None, optimizer='adam', optimizer_params={'lr': 0.01}, verbose=True)
    model.fit(X['train'])

    X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    entities_subset = np.concatenate([X['test'][::1000, 0], X['test'][::1000, 2]], 0)

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o',
                                 entities_subset=entities_subset)
    ranks = ranks.reshape(-1)
    assert (np.sum(ranks > len(entities_subset)) == 0)


def test_evaluate_performance_ranking_against_shuffled_all_entities():
    """ Compares mrr of test set by using default protocol against all entities vs 
        mrr of corruptions generated by corrupting using entities_subset = all entities shuffled
    """
    import random
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=1, k=20, eta=10, loss='nll',
                    regularizer=None, optimizer='adam', optimizer_params={'lr': 0.01}, verbose=True)
    model.fit(X['train'])

    X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    entities_subset = random.shuffle(list(model.ent_to_idx.keys()))

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks_all = evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o')

    ranks_suffled_ent = evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o',
                                             entities_subset=entities_subset)
    assert (mrr_score(ranks_all) == mrr_score(ranks_suffled_ent))


def test_evaluate_performance_default_protocol_without_filter():
    wn18 = load_wn18()

    model = TransE(batches_count=10, seed=0, epochs=1,
                   k=50, eta=10, verbose=True,
                   embedding_model_params={'normalize_ent_emb': False, 'norm': 1},
                   loss='self_adversarial', loss_params={'margin': 1, 'alpha': 0.5},
                   optimizer='adam',
                   optimizer_params={'lr': 0.0005})

    model.fit(wn18['train'])

    from bigraph.evaluation import evaluate_performance
    ranks_sep = []
    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, verbose=True, corrupt_side='o')

    ranks_sep.extend(ranks)
    from bigraph.evaluation import evaluate_performance

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, verbose=True, corrupt_side='s')
    ranks_sep.extend(ranks)
    print('----------EVAL WITHOUT FILTER-----------------')
    print('----------Subj and obj corrupted separately-----------------')
    mr_sep = mr_score(ranks_sep)
    print('MAR:', mr_sep)
    print('Mrr:', mrr_score(ranks_sep))
    print('hits10:', hits_at_n_score(ranks_sep, 10))
    print('hits3:', hits_at_n_score(ranks_sep, 3))
    print('hits1:', hits_at_n_score(ranks_sep, 1))

    from bigraph.evaluation import evaluate_performance

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, verbose=True, corrupt_side='s,o')
    print('----------corrupted with default protocol-----------------')
    mr_joint = mr_score(ranks)
    mrr_joint = mrr_score(ranks)
    print('MAR:', mr_joint)
    print('Mrr:', mrr_score(ranks))
    print('hits10:', hits_at_n_score(ranks, 10))
    print('hits3:', hits_at_n_score(ranks, 3))
    print('hits1:', hits_at_n_score(ranks, 1))

    np.testing.assert_equal(mr_sep, mr_joint)
    assert (mrr_joint is not np.Inf)


def test_evaluate_performance_default_protocol_with_filter():
    wn18 = load_wn18()

    X_filter = np.concatenate((wn18['train'], wn18['valid'], wn18['test']))

    model = TransE(batches_count=10, seed=0, epochs=1,
                   k=50, eta=10, verbose=True,
                   embedding_model_params={'normalize_ent_emb': False, 'norm': 1},
                   loss='self_adversarial', loss_params={'margin': 1, 'alpha': 0.5},
                   optimizer='adam',
                   optimizer_params={'lr': 0.0005})

    model.fit(wn18['train'])

    from bigraph.evaluation import evaluate_performance
    ranks_sep = []
    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, X_filter, verbose=True, corrupt_side='o')

    ranks_sep.extend(ranks)
    from bigraph.evaluation import evaluate_performance

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, X_filter, verbose=True, corrupt_side='s')
    ranks_sep.extend(ranks)
    print('----------EVAL WITH FILTER-----------------')
    print('----------Subj and obj corrupted separately-----------------')
    mr_sep = mr_score(ranks_sep)
    print('MAR:', mr_sep)
    print('Mrr:', mrr_score(ranks_sep))
    print('hits10:', hits_at_n_score(ranks_sep, 10))
    print('hits3:', hits_at_n_score(ranks_sep, 3))
    print('hits1:', hits_at_n_score(ranks_sep, 1))

    from bigraph.evaluation import evaluate_performance

    from bigraph.evaluation import hits_at_n_score, mrr_score, mr_score
    ranks = evaluate_performance(wn18['test'][::100], model, X_filter, verbose=True, corrupt_side='s,o')
    print('----------corrupted with default protocol-----------------')
    mr_joint = mr_score(ranks)
    mrr_joint = mrr_score(ranks)
    print('MAR:', mr_joint)
    print('Mrr:', mrr_joint)
    print('hits10:', hits_at_n_score(ranks, 10))
    print('hits3:', hits_at_n_score(ranks, 3))
    print('hits1:', hits_at_n_score(ranks, 1))

    np.testing.assert_equal(mr_sep, mr_joint)
    assert (mrr_joint is not np.Inf)


def test_evaluate_performance_so_side_corruptions_with_filter():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=5, k=200, eta=10, loss='nll',
                    regularizer=None, optimizer='adam', optimizer_params={'lr': 0.01}, verbose=True)
    model.fit(X['train'])

    ranks = evaluate_performance(X['test'][::20], model=model, verbose=True, corrupt_side='s+o')
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("ranks: %s" % ranks)
    print("MRR: %f" % mrr)
    print("Hits@10: %f" % hits_10)
    assert (mrr is not np.Inf)


@pytest.mark.skip(reason="CircleCI free tier upper bound investigation")
def test_evaluate_performance_so_side_corruptions_without_filter():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=5, k=200, eta=10, loss='nll',
                    regularizer=None, optimizer='adam', optimizer_params={'lr': 0.01}, verbose=True)
    model.fit(X['train'])

    X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    ranks = evaluate_performance(X['test'][::20], model, X_filter, verbose=True, corrupt_side='s+o')
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("ranks: %s" % ranks)
    print("MRR: %f" % mrr)
    print("Hits@10: %f" % hits_10)
    assert (mrr is not np.Inf)