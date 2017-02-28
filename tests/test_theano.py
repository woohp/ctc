from __future__ import absolute_import, print_function
import unittest
import numpy as np
import theano
import theano.tensor as T

from ctc.th import ctc_equals, ctc_cer, ctc_loss, ctc_loss_only


class TestCTCTheano(unittest.TestCase):
    def test_equals_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        label_lengths = T.fvector()
        eq = ctc_equals(y, l, label_lengths)
        eq_function = theano.function([y, l, label_lengths], eq)

        y_ = np.array([
            [[0, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)
        label_lengths_ = np.array([1], dtype=np.float32)

        self.assertEqual(eq_function(y_, l_, label_lengths_), [1])

    def test_loss_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        loss = ctc_loss(y, l)
        loss_function = theano.function([y, l], loss)
        grad_function = theano.function([y, l], T.grad(loss.sum(), y))

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]],
            [[0.01, 0.01, 0.01, 0.97],
             [0.01, 0.97, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0], [1]], dtype=np.float32)

        expected_gradient = np.array([
            [[-1.03061, 0., 0., -1.0203],
             [-1.03071, 0., 0., -0.02082],
             [-1.03061, 0., 0., -1.0203]],
            [[0., -1.03061, 0., -1.0203],
             [0., -1.03071, 0., -0.02082],
             [0., -1.03061, 0., -1.0203]],
        ], dtype=np.float32)

        losses = loss_function(y_, l_)
        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(losses[0], 0.07066, 5)
        self.assertAlmostEqual(losses[1], 0.07066, 5)
        self.assertAlmostEqual(np.abs(grad_function(y_, l_) - expected_gradient).sum(), 0, 3)

    def test_loss_only_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        loss = ctc_loss_only(y, l)
        loss_function = theano.function([y, l], loss)
        with self.assertRaises(Exception):
            theano.function([y, l], T.grad(loss.sum(), y))

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)

        loss = loss_function(y_, l_)
        self.assertAlmostEqual(float(loss), 0.07066, 5)

    def test_character_error_rate_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        label_lengths = T.fvector()
        cer = ctc_cer(y, l, label_lengths)
        cer_function = theano.function([y, l, label_lengths], cer)

        y_ = np.array([
            [[0, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 1]]
        ], dtype=np.float32)
        l_ = np.array([[0, 1]], dtype=np.float32)
        label_lengths_ = np.array([2], dtype=np.float32)

        self.assertAlmostEqual(cer_function(y_, l_, label_lengths_)[0], 0.5)
