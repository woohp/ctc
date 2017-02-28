from __future__ import absolute_import, print_function
import unittest
import numpy as np

import ctc


np.set_printoptions(precision=5, suppress=True, linewidth=1000)


class TestCTC(unittest.TestCase):
    def setUp(self):
        self.n_classes = 3
        self.l = np.array([[0, 1, 2, 1, 0]], dtype=np.int32)
        self.y = np.array([[
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.97],
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ]], dtype=np.float32)

        self.expected_loss = 3.72168573746
        self.expected_gradient = np.array([
            [[-3.34564171e+01, -0.00000000e+00, -0.00000000e+00, -6.86016142e-01],
             [-3.37871819e+01, -7.03715952e-03, -0.00000000e+00, -6.82533741e-01],
             [-3.31080933e+01, -6.89533591e-01, -1.47262097e-06, -6.82498634e-01],
             [-3.37802559e-01, -1.02722812e+00, -2.15388747e-04, -2.08568815e-02],
             [-7.00052915e-05, -1.35113621e+00, -1.01695991e+00, -1.00651371e+00],
             [-7.21365154e-07, -3.94263444e-03, -1.03067279e+00, -2.07894091e-02],
             [-4.03621456e-08, -1.02036762e+00, -1.02384794e+00, -1.00985336e+00],
             [-2.12444225e-04, -1.03070843e+00, -2.51622085e-04, -2.08220743e-02],
             [-1.02019310e+00, -1.02044320e+00, -7.76845411e-07, -1.00989020e+00],
             [-1.03071094e+00, -2.15784064e-04, -0.00000000e+00, -2.08213553e-02],
             [-1.03082049e+00, -0.00000000e+00, -0.00000000e+00, -1.02030087e+00]]
        ], dtype=np.float32)

    def test_ctc(self):
        y = self.y
        l = self.l

        loss, grad = ctc.loss(y, l)

        self.assertAlmostEqual(float(loss), self.expected_loss, 5)
        self.assertTrue(np.abs(grad - self.expected_gradient).sum() < 2e-5)

    def test_ctc_loss_only(self):
        y = self.y
        l = self.l

        loss = ctc.loss_only(y, l)

        self.assertAlmostEqual(float(loss), self.expected_loss, 5)

    def test_ctc_batch(self):
        y = np.array([
            [[0.1, 0.1, 0.1, 0.9],
             [0.1, 0.9, 0.1, 0.1],
             [0.1, 0.1, 0.1, 0.9]],
            [[0.1, 0.1, 0.1, 0.9],
             [0.1, 0.1, 0.9, 0.1],
             [0.1, 0.1, 0.1, 0.9]],
        ], dtype=np.float32)
        l = np.array([[1], [2]], dtype=np.int32)

        loss, grad = ctc.loss(y, l)
        self.assertEqual(loss[0], loss[1])

    def test_ctc_loss_only_batch(self):
        y = np.array([
            [[0.1, 0.1, 0.1, 0.9],
             [0.1, 0.9, 0.1, 0.1],
             [0.1, 0.1, 0.1, 0.9]],
            [[0.1, 0.1, 0.1, 0.9],
             [0.1, 0.1, 0.9, 0.1],
             [0.1, 0.1, 0.1, 0.9]],
        ], dtype=np.float32)
        l = np.array([[1], [2]], dtype=np.int32)

        loss = ctc.loss_only(y, l)
        self.assertEqual(loss[0], loss[1])

    def test_ctc_long_sequence(self):
        l = np.array([0, 1] * 120, dtype=np.int32)[None, :]
        y = np.array([
            [0.01, 0.01, 0.01, 0.97],
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.97],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.97]
        ] * 100, dtype=np.float32)[None, :, :]

        loss, grad = ctc.loss(y, l)

        self.assertTrue(np.isfinite(grad.sum()))

    def test_bad_inputs(self):
        l = np.array([0], dtype=np.int32)[None, :]
        y = np.array([
            [0, 0, 0, 30],
            [30, 0, 0, 0],
            [0, 0, 0, 30],
        ], dtype=np.float32)[None, :, :]

        loss, grad = ctc.loss(y, l)

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(np.abs(grad).sum()))

    def test_bad_inputs_2(self):
        l = np.array([0, 1], dtype=np.int32)[None, :]
        y = np.array([
            [0, 0, 0, 30],
            [30, 0, 0, 0],
            [0, 0, 0, 30],
            [0, 0, 0, 30],
        ], dtype=np.float32)[None, :, :]

        loss, grad = ctc.loss(y, l)

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(np.abs(grad).sum()))

    def test_bad_inputs_3(self):
        l = np.array([0], dtype=np.int32)[None, :]
        y = np.array([
            [0, 0, 0, 30],
            [0, 30, 0, 0],
            [0, 0, 0, 30],
        ] * 100, dtype=np.float32)[None, :, :]

        loss, grad = ctc.loss(y, l)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(np.abs(grad).sum()))

    def test_bad_inputs_4(self):
        y = np.array([
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)[None, :, :]
        l = np.array([[2, 0, 2, 2, 2, 2]], dtype=np.int32)

        loss, grad = ctc.loss(y, l)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(np.abs(grad).sum()))

    def test_decode(self):
        l = np.array([
            [0, 0, 0.1, 0.9],
            [1, 0, 0, 0],
            [0.7, 0, 0.1, 0.2],
            [0, 0, 0, 1],
            [0.3, 0.4, 0.3, 0],
            [0, 0, 0, 1],
            [0.1, 0.2, 0.3, 0.4],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.assertEqual(ctc.decode(l).tolist(), [0, 1, 2])

    def test_equals(self):
        y = np.array([
            [
                [0, 0, 0.1, 0.9],
                [1, 0, 0, 0],
                [0.7, 0, 0.1, 0.2],
                [0, 0, 0, 1],
                [0.3, 0.4, 0.3, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                [0.8, 0, 0, 0.2],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ], dtype=np.float32)

        y_true = np.array([
            [0, 1, 2],
            [0, 1, 1],
        ], dtype=np.int32)

        label_lengths = np.array([3, 3], dtype=np.int32)

        self.assertEqual(ctc.equals(y, y_true, label_lengths).tolist(), [True, False])

    def test_equals_blank(self):
        y = np.array([[
            [0, 0, 0.1, 0.9],
            [0, 0, 0.1, 0.9],
            [0, 0, 0.1, 0.9],
            [0, 0, 0.1, 0.9],
            [0, 0, 0.1, 0.9],
            [0, 0, 0.1, 0.9]]
        ], dtype=np.float32)

        y_true = np.array([[0, 1, 2]], dtype=np.int32)

        label_lengths = np.array([3], dtype=np.int32)

        self.assertEqual(ctc.equals(y, y_true, label_lengths).tolist(), [False])

    def test_edit_distance(self):
        y = np.array([
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            [
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ]
        ], dtype=np.float32)

        y_true = np.array([
            [0, 1, 2],
            [0, 1, 0],
            [2, 2, 1],
            [0, 1, 2],
            [1, 2, 1],
        ], dtype=np.int32)

        label_lengths = np.array([3, 3, 3, 3, 3], dtype=np.int32)

        self.assertEqual(ctc.edit_distance(y, y_true, label_lengths).tolist(), [0, 1, 2, 1, 3])

    def test_character_error_rate(self):
        y = np.array([
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            [
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ]
        ], dtype=np.float32)

        y_true = np.array([
            [0, 1, 2],
            [0, 1, 0],
            [2, 2, 1],
            [0, 1, 2],
            [1, 2, 1],
        ], dtype=np.int32)

        label_lengths = np.array([3, 3, 3, 3, 3], dtype=np.int32)

        expected = np.array([0, 1.0 / 3, 2.0 / 3, 1.0 / 3, 1.0])
        cer = ctc.character_error_rate(y, y_true, label_lengths)
        self.assertTrue(np.abs(cer - expected).sum() < 1e-5)

    def test_wide(self):
        l = np.array([[79], [78]], dtype=np.int32)
        y = np.array([
            [[0.00001] * 80 + [0.9999],
             [0.9999] + [0.00001] * 80,
             [0.00001] * 80 + [0.9999]],

            [[0.00001] * 80 + [0.9999],
             [0.9999] + [0.00001] * 80,
             [0.00001] * 80 + [0.9999]],
        ], dtype=np.float32)

        loss, grad = ctc.loss(y, l)
