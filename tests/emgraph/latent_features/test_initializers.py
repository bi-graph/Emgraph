import os

import numpy as np
import tensorflow as tf

from emgraph.initializers._initializer_constants import INITIALIZER_REGISTRY
from emgraph.utils.misc import make_variable
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test_random_normal():
    """Random normal initializer test
    """
    tf.random.set_seed(0)
    rnormal_class = INITIALIZER_REGISTRY['normal']
    rnormal_obj = rnormal_class({"mean": 0.5, "std": 0.1})
    tf_init = rnormal_obj.get_entity_initializer(init_type='tf')
    var1 = make_variable(shape=(100, 10), initializer=tf_init, name="var1")
    np_var = rnormal_obj.get_entity_initializer(100, 10, init_type='np')
    tf_var = tf.convert_to_tensor(var1)
    assert (np.round(np.mean(np_var), 1) == np.round(np.mean(tf_var), 1))
    assert (np.round(np.std(np_var), 1) == np.round(np.std(tf_var), 1))


def test_glorot_normal():
    """GlorotUniform normal initializer test
    """
    tf.random.set_seed(0)
    xnormal_class = INITIALIZER_REGISTRY['glorot_uniform']
    xnormal_obj = xnormal_class({"uniform": False})
    tf_init = xnormal_obj.get_entity_initializer(init_type='tf')
    var1 = make_variable(shape=(200, 10), initializer=tf_init, name="var1")
    tf_var = tf.convert_to_tensor(var1)
    np_var = xnormal_obj.get_entity_initializer(200, 10, init_type='np')
    # print(np.mean(np_var), np.std(np_var))
    # print(np.mean(tf_var), np.std(tf_var))
    assert (np.round(np.mean(np_var), 2) == np.round(np.mean(tf_var), 2))
    assert (np.round(np.std(np_var), 2) == np.round(np.std(tf_var), 2))


def test_glorot_uniform():
    """GlorotUniform uniform initializer test
    """
    tf.random.set_seed(0)
    xuniform_class = INITIALIZER_REGISTRY['glorot_uniform']
    xuniform_obj = xuniform_class({"uniform": True})
    tf_init = xuniform_obj.get_entity_initializer(init_type='tf')
    var1 = make_variable(shape=(20, 100), initializer=tf_init, name="var1")
    tf_var = tf.convert_to_tensor(var1)
    np_var = xuniform_obj.get_entity_initializer(20, 100, init_type='np')
    # print(np.min(np_var), np.max(np_var))
    # print(np.min(tf_var), np.max(tf_var))
    assert (np.round(np.min(np_var), 2) == np.round(np.min(tf_var), 2))
    assert (np.round(np.max(np_var), 2) == np.round(np.max(tf_var), 2))


def test_random_uniform():
    """Random uniform initializer test
    """
    tf.random.set_seed(0)
    runiform_class = INITIALIZER_REGISTRY['uniform']
    runiform_obj = runiform_class({"low": 0.1, "high": 0.4})
    tf_init = runiform_obj.get_entity_initializer(init_type='tf')
    var1 = make_variable(shape=(100, 10), initializer=tf_init, name="var1")
    tf_var = tf.convert_to_tensor(var1)
    np_var = runiform_obj.get_entity_initializer(100, 10, init_type='np')
    # print(np.min(np_var), np.max(np_var))
    # print(np.min(tf_var), np.max(tf_var))
    assert (np.round(np.min(np_var), 2) == np.round(np.min(tf_var), 2))
    assert (np.round(np.max(np_var), 2) == np.round(np.max(tf_var), 2))


def test_constant():
    """Constant initializer test
    """
    tf.random.set_seed(117)
    runiform_class = INITIALIZER_REGISTRY['constant']
    ent_init = np.random.normal(1, 1, size=(300, 30))
    rel_init = np.random.normal(2, 2, size=(10, 30))
    runiform_obj = runiform_class({"entity": ent_init, "relation": rel_init})
    var1 = make_variable(
        shape=(300, 30),
        initializer=runiform_obj.get_entity_initializer(300, 30, init_type='tf'),
        name="ent_var"
        )
    var2 = make_variable(
        shape=(10, 30),
        initializer=runiform_obj.get_relation_initializer(10, 30, init_type='tf'),
        name="rel_var"
        )
    tf_var1 = tf.convert_to_tensor(var1)
    tf_var2 = tf.convert_to_tensor(var2)

    np_var1 = runiform_obj.get_entity_initializer(300, 30, init_type='np')
    np_var2 = runiform_obj.get_relation_initializer(10, 30, init_type='np')
    assert (np.round(np.mean(tf_var1), 0) == np.round(np.mean(np_var1), 0))
    assert (np.round(np.std(tf_var1), 0) == np.round(np.std(np_var1), 0))
    assert (np.round(np.mean(tf_var2), 0) == np.round(np.mean(np_var2), 0))
    assert (np.round(np.std(tf_var2), 0) == np.round(np.std(np_var2), 0))
