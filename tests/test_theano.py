import unittest
import numpy as np
import theano
import theano.tensor as T

from ctc.th import ctc_equals, ctc_cost, ctc_cost_only


class TestCTCTheano(unittest.TestCase):
    def test_equals_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        eq = ctc_equals(y, l)
        eq_function = theano.function([y, l], eq)

        y_ = np.array([
            [[0, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)

        self.assertEqual(eq_function(y_, l_), [1])

    def test_cost_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        cost = ctc_cost(y, l)
        cost_function = theano.function([y, l], cost)
        grad_function = theano.function([y, l], T.grad(cost.sum(), y))

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)

        expected_gradient = np.array([[
            [-1.03061, 0., 0., -1.0203],
            [-1.03071, 0., 0., -0.02082],
            [-1.03061, 0., 0., -1.0203]
        ]], dtype=np.float32)

        self.assertAlmostEqual(cost_function(y_, l_), 0.07066, 5)
        self.assertAlmostEqual(np.abs(grad_function(y_, l_) - expected_gradient).sum(), 0, 3)

    def test_cost_only_op(self):
        y = T.ftensor3()
        l = T.fmatrix()
        cost = ctc_cost_only(y, l)
        cost_function = theano.function([y, l], cost)
        with self.assertRaises(Exception):
            theano.function([y, l], T.grad(cost.sum(), y))

        y_ = np.array([
            [[0.01, 0.01, 0.01, 0.97],
             [0.97, 0.01, 0.01, 0.01],
             [0.01, 0.01, 0.01, 0.97]]
        ], dtype=np.float32)
        l_ = np.array([[0]], dtype=np.float32)

        self.assertAlmostEqual(cost_function(y_, l_), 0.07066, 5)
