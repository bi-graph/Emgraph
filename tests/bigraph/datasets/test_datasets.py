from bigraph.datasets import load_wn18, load_fb15k, load_fb15k_237, load_yago3_10, load_wn18rr, load_wn11, \
    load_fb13, load_onet20k, load_ppi5k, load_nl27k, load_cn15k, OneToNDatasetAdapter, load_from_ntriples
from bigraph.datasets.datasets import _clean_data
import os
import numpy as np
import pytest


def test_clean_data():
    X = {
        'train': np.array([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]),
        'valid': np.array([['a', 'b', 'c'], ['x', 'e', 'f'], ['g', 'a', 'i'], ['j', 'k', 'y']]),
        'test': np.array([['a', 'b', 'c'], ['d', 'e', 'x'], ['g', 'b', 'i'], ['y', 'k', 'l']]),
    }

    clean_X, valid_idx, test_idx = _clean_data(X, return_idx=True)

    np.testing.assert_array_equal(clean_X['train'], X['train'])
    np.testing.assert_array_equal(clean_X['valid'], np.array([['a', 'b', 'c']]))
    np.testing.assert_array_equal(clean_X['test'], np.array([['a', 'b', 'c'], ['g', 'b', 'i']]))
    np.testing.assert_array_equal(valid_idx, np.array([True, False, False, False]))
    np.testing.assert_array_equal(test_idx, np.array([True, False, True, False]))


def test_load_wn18():
    wn18 = load_wn18()
    assert len(wn18['train']) == 141442
    assert len(wn18['valid']) == 5000
    assert len(wn18['test']) == 5000

    ent_train = np.union1d(np.unique(wn18["train"][:, 0]), np.unique(wn18["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18["valid"][:, 0]), np.unique(wn18["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18["test"][:, 0]), np.unique(wn18["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18["train"][:, 1]), np.unique(wn18["train"][:, 1])),
                              np.unique(wn18["train"][:, 1]))

    assert len(distinct_ent) == 40943
    assert len(distinct_rel) == 18


def test_reciprocals():
    """Test for reciprocal relations
    """
    # Create dataset with reciprocal relations and test if the are added
    fb15k = load_fb15k(add_reciprocal_rels=True)
    train_reciprocal = fb15k['train']
    triple = train_reciprocal[0]
    reciprocal_triple = train_reciprocal[train_reciprocal.shape[0] // 2]
    assert (triple[0] == reciprocal_triple[2])
    assert (triple[2] == reciprocal_triple[0])
    assert (triple[1] + '_reciprocal' == reciprocal_triple[1])

    # create the same dataset without reciprocals. Now the number of triples should be half of prev
    fb15k = load_fb15k(add_reciprocal_rels=False)
    assert (fb15k['train'].shape[0] == train_reciprocal.shape[0] // 2)


def test_load_fb15k():
    fb15k = load_fb15k()
    assert len(fb15k['train']) == 483142
    assert len(fb15k['valid']) == 50000
    assert len(fb15k['test']) == 59071

    # ent_train = np.union1d(np.unique(fb15k["train"][:,0]), np.unique(fb15k["train"][:,2]))
    # ent_valid = np.union1d(np.unique(fb15k["valid"][:,0]), np.unique(fb15k["valid"][:,2]))
    # ent_test = np.union1d(np.unique(fb15k["test"][:,0]), np.unique(fb15k["test"][:,2]))
    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(fb15k["train"][:,1]), np.unique(fb15k["train"][:,1])),
    #                           np.unique(fb15k["train"][:,1]))

    # assert len(distinct_ent) == 14951
    # assert len(distinct_rel) == 1345


def test_load_fb15k_237():
    fb15k_237 = load_fb15k_237()
    assert len(fb15k_237['train']) == 272115

    # - 9 because 9 triples containing unseen entities are removed
    assert len(fb15k_237['valid']) == 17535 - 9

    # - 28 because 28 triples containing unseen entities are removed
    assert len(fb15k_237['test']) == 20466 - 28


def test_yago_3_10():
    yago_3_10 = load_yago3_10()
    assert len(yago_3_10['train']) == 1079040
    assert len(yago_3_10['valid']) == 5000 - 22
    assert len(yago_3_10['test']) == 5000 - 18

    # ent_train = np.union1d(np.unique(yago_3_10["train"][:,0]), np.unique(yago_3_10["train"][:,2]))
    # ent_valid = np.union1d(np.unique(yago_3_10["valid"][:,0]), np.unique(yago_3_10["valid"][:,2]))
    # ent_test = np.union1d(np.unique(yago_3_10["test"][:,0]), np.unique(yago_3_10["test"][:,2]))

    # assert len(set(ent_valid) - set(ent_train)) == 22
    # assert len (set(ent_test) - ((set(ent_valid) & set(ent_train)) | set(ent_train))) == 18

    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(yago_3_10["train"][:,1]), np.unique(yago_3_10["train"][:,1])),
    #                           np.unique(yago_3_10["train"][:,1]))

    # assert len(distinct_ent) == 123182
    # assert len(distinct_rel) == 37


def test_wn18rr():
    wn18rr = load_wn18rr()

    ent_train = np.union1d(np.unique(wn18rr["train"][:, 0]), np.unique(wn18rr["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18rr["valid"][:, 0]), np.unique(wn18rr["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18rr["test"][:, 0]), np.unique(wn18rr["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18rr["train"][:, 1]), np.unique(wn18rr["train"][:, 1])),
                              np.unique(wn18rr["train"][:, 1]))

    assert len(wn18rr['train']) == 86835

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['valid']) == 3034 - 210

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['test']) == 3134 - 210


def test_wn11():
    wn11 = load_wn11(clean_unseen=False)
    assert len(wn11['train']) == 110361
    assert len(wn11['valid']) == 5215
    assert len(wn11['test']) == 21035
    assert len(wn11['valid_labels']) == 5215
    assert len(wn11['test_labels']) == 21035
    assert sum(wn11['valid_labels']) == 2606
    assert sum(wn11['test_labels']) == 10493

    wn11 = load_wn11(clean_unseen=True)
    assert len(wn11['train']) == 110361
    assert len(wn11['valid']) == 5215 - 338
    assert len(wn11['test']) == 21035 - 1329
    assert len(wn11['valid_labels']) == 5215 - 338
    assert len(wn11['test_labels']) == 21035 - 1329
    assert sum(wn11['valid_labels']) == 2409
    assert sum(wn11['test_labels']) == 9706


def test_fb13():
    fb13 = load_fb13(clean_unseen=False)
    assert len(fb13['train']) == 316232
    assert len(fb13['valid']) == 5908 + 5908
    assert len(fb13['test']) == 23733 + 23731
    assert len(fb13['valid_labels']) == 5908 + 5908
    assert len(fb13['test_labels']) == 23733 + 23731
    assert sum(fb13['valid_labels']) == 5908
    assert sum(fb13['test_labels']) == 23733

    fb13 = load_fb13(clean_unseen=True)
    assert len(fb13['train']) == 316232
    assert len(fb13['valid']) == 5908 + 5908
    assert len(fb13['test']) == 23733 + 23731
    assert len(fb13['valid_labels']) == 5908 + 5908
    assert len(fb13['test_labels']) == 23733 + 23731
    assert sum(fb13['valid_labels']) == 5908
    assert sum(fb13['test_labels']) == 23733


def test_onet20k():
    onet = load_onet20k()
    assert len(onet['train']) == 461932
    assert len(onet['valid']) == 850
    assert len(onet['test']) == 2000
    assert len(onet['train_numeric_values']) == 461932
    assert len(onet['valid_numeric_values']) == 850
    assert len(onet['test_numeric_values']) == 2000


def test_nl27k():
    nl27k = load_nl27k()
    assert len(nl27k['train']) == 149100
    assert len(nl27k['valid']) == 12274
    assert len(nl27k['test']) == 14026
    assert len(nl27k['train_numeric_values']) == 149100
    assert len(nl27k['valid_numeric_values']) == 12274
    assert len(nl27k['test_numeric_values']) == 14026

    nl27k = load_nl27k(clean_unseen=False)
    assert len(nl27k['train']) == 149100
    assert len(nl27k['valid']) == 12274 + 4
    assert len(nl27k['test']) == 14026 + 8
    assert len(nl27k['train_numeric_values']) == 149100
    assert len(nl27k['valid_numeric_values']) == 12274 + 4
    assert len(nl27k['test_numeric_values']) == 14026 + 8


def test_ppi5k():
    ppi5k = load_ppi5k()
    assert len(ppi5k['train']) == 230929
    assert len(ppi5k['valid']) == 19017
    assert len(ppi5k['test']) == 21720
    assert len(ppi5k['train_numeric_values']) == 230929
    assert len(ppi5k['valid_numeric_values']) == 19017
    assert len(ppi5k['test_numeric_values']) == 21720


def test_cn15k():
    cn15k = load_cn15k()
    assert len(cn15k['train']) == 199417
    assert len(cn15k['valid']) == 16829
    assert len(cn15k['test']) == 19224
    assert len(cn15k['train_numeric_values']) == 199417
    assert len(cn15k['valid_numeric_values']) == 16829
    assert len(cn15k['test_numeric_values']) == 19224


def test_load_from_ntriples(request):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'tests', 'ampligraph', 'datasets')
    X = load_from_ntriples('', 'test_triples.nt', data_home=path)
    assert X.shape == (3, 3)
    assert len(np.unique(X.flatten())) == 6


