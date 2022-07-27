import numpy as np
import tensorflow as tf

from emgraph.regularizers._regularizer_constants import REGULARIZER_REGISTRY


def test_l1_regularizer():
    l1_class = REGULARIZER_REGISTRY["LP"]
    p1 = tf.constant([1, -1, 1], dtype=tf.float32)
    p2 = tf.constant([2, -2, 2], dtype=tf.float32)
    params = [p1, p2]
    lambda_1 = 1.0
    lambda_2 = [2.0, 3.0]
    l1_obj1 = l1_class({"lambda": lambda_1, "p": 1})
    l1_obj2 = l1_class({"lambda": lambda_2, "p": 1})

    with tf.compat.v1.Session() as sess:
        out = sess.run(l1_obj1.apply([p1, p2]))
        np.testing.assert_array_equal(out, 9.0)
        out = sess.run(l1_obj2.apply([p1, p2]))
        np.testing.assert_array_equal(out, 24.0)


def test_l2_regularizer():
    l2_class = REGULARIZER_REGISTRY["LP"]
    p1 = tf.constant([1, -1, 1], dtype=tf.float32)
    p2 = tf.constant([2, -2, 2], dtype=tf.float32)
    params = [p1, p2]
    lambda_1 = 1.0
    lambda_2 = [2.0, 3.0]
    l2_obj1 = l2_class({"lambda": lambda_1, "p": 2})
    l2_obj2 = l2_class({"lambda": lambda_2, "p": 2})

    with tf.compat.v1.Session() as sess:
        out = sess.run(l2_obj1.apply([p1, p2]))
        np.testing.assert_array_equal(out, 15.0)
        out = sess.run(l2_obj2.apply([p1, p2]))
        np.testing.assert_array_equal(out, 42.0)
