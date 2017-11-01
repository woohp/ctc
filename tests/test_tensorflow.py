from __future__ import absolute_import, print_function
import unittest
import numpy as np
import tensorflow as tf

from ctc.tf import ctc_equals, ctc_cer


class TestCTCTheano(unittest.TestCase):
    def test_equals_op(self):
        y = tf.placeholder(name='y', shape=(None, None, None), dtype=tf.float32)
        l = tf.placeholder(name='l', dtype=tf.int32)
        label_lengths = tf.placeholder(name='label_lengths', dtype=tf.int32)
        eq = ctc_equals(y, l, label_lengths)

        y_ = np.array([
            [[0, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)
        label_lengths_ = np.array([[1]], dtype=np.float32)

        with tf.Session('') as session:
            equals = session.run(eq, {y: y_, l: l_, label_lengths: label_lengths_})
        self.assertEqual(equals, [1])

    def test_character_error_rate_op(self):
        y = tf.placeholder(name='y', dtype=tf.float32)
        l = tf.placeholder(name='l', dtype=tf.int32)
        label_lengths = tf.placeholder(name='label_lengths', dtype=tf.int32)
        cer = ctc_cer(y, l, label_lengths)

        y_ = np.array([
            [[0, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 1]]
        ], dtype=np.float32)
        l_ = np.array([[0, 1]], dtype=np.float32)
        label_lengths_ = np.array([[2]], dtype=np.float32)

        with tf.Session('') as session:
            err = session.run(cer, {y: y_, l: l_, label_lengths: label_lengths_})
        self.assertAlmostEqual(err[0], 0.5)
