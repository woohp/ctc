from __future__ import absolute_import, print_function
import unittest
import numpy as np
import tensorflow as tf

from ctc.tf import ctc_loss, ctc_loss_only, ctc_equals, ctc_cer


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

    def test_loss_op(self):
        y = tf.placeholder(name='y', shape=(None, None, None), dtype=tf.float32)
        l = tf.placeholder(name='l', shape=(None, None), dtype=tf.int32)
        loss = ctc_loss(y, l)
        grad = tf.gradients([loss], [y])

        self.assertEqual(len(loss.get_shape()._dims), 1)

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]],
            [[0.01, 0.01, 0.01, 0.97],
             [0.01, 0.97, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0], [1]], dtype=np.int32)

        with tf.Session('') as session:
            losses = session.run(loss, {y: y_, l: l_})
            gradients = session.run(grad, {y: y_, l: l_})

        expected_gradient = np.array([
            [[-1.03061, 0., 0., -1.0203],
             [-1.03071, 0., 0., -0.02082],
             [-1.03061, 0., 0., -1.0203]],
            [[0., -1.03061, 0., -1.0203],
             [0., -1.03071, 0., -0.02082],
             [0., -1.03061, 0., -1.0203]],
        ], dtype=np.float32)

        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(losses[0], 0.07066, 5)
        self.assertAlmostEqual(losses[1], 0.07066, 5)
        self.assertAlmostEqual(np.abs(gradients - expected_gradient).sum(), 0, 3)

    def test_loss_only_op(self):
        y = tf.placeholder(name='y', dtype=tf.float32)
        l = tf.placeholder(name='l', dtype=tf.int32)
        loss = ctc_loss_only(y, l)
        with self.assertRaises(Exception):
            tf.gradients([loss], [y])

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)

        with tf.Session('') as session:
            loss = session.run(loss, {y: y_, l: l_})
        self.assertAlmostEqual(float(loss), 0.07066, 5)

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
