from __future__ import absolute_import, print_function
import numpy as np
import theano
import theano.tensor as T

import ctc


class CTCEqualsOp(theano.Op):
    def make_node(self, y_pred, y_true, label_lengths):
        y_pred = T.as_tensor_variable(y_pred)
        y_true = T.as_tensor_variable(y_true)
        label_lengths = T.as_tensor_variable(label_lengths)
        assert y_pred.ndim == 3
        assert y_true.ndim == 2
        assert label_lengths.ndim == 1
        return theano.Apply(self, [y_pred, y_true, label_lengths], [T.bvector()])

    def perform(self, node, inputs, outputs):
        y_pred, y_true, label_lengths = inputs
        y_true = y_true.astype(np.int32)
        label_lengths = label_lengths.astype(np.int32)

        outputs[0][0] = ctc.equals(y_pred, y_true, label_lengths).astype(np.int8)


class CTCCharacterErrorRate(theano.Op):
    def make_node(self, y_pred, y_true, label_lengths):
        y_pred = T.as_tensor_variable(y_pred)
        y_true = T.as_tensor_variable(y_true)
        label_lengths = T.as_tensor_variable(label_lengths)
        assert y_pred.ndim == 3
        assert y_true.ndim == 2
        assert label_lengths.ndim == 1
        return theano.Apply(self, [y_pred, y_true, label_lengths], [T.fvector()])

    def perform(self, node, inputs, outputs):
        y_pred, y_true, label_lengths = inputs
        y_true = y_true.astype(np.int32)
        label_lengths = label_lengths.astype(np.int32)

        outputs[0][0] = ctc.character_error_rate(y_pred, y_true, label_lengths)


class CTCLossMixin(object):
    def make_node(self, y, l):
        y = T.as_tensor_variable(y)
        l = T.as_tensor_variable(l)
        assert y.ndim == 3
        assert l.ndim == 2
        return theano.Apply(self, [y, l], [T.fvector(), T.ftensor3()])

    def grad(self, inputs, output_grads):
        y, l = inputs
        grad = self(y, l)[1]
        return [grad * output_grads[0][:, None, None],
                theano.gradient.grad_not_implemented(l, 1, '')]


class CTCLossOpCpu(CTCLossMixin, theano.Op):
    def perform(self, node, inputs, outputs):
        y, l = inputs
        l = l.astype(np.int32)

        outputs[0][0], outputs[1][0] = ctc.loss(y, l)


class CTCLossOnlyOpCPU(theano.Op):
    def make_node(self, y, l):
        y = T.as_tensor_variable(y)
        l = T.as_tensor_variable(l)
        assert y.ndim == 3
        assert l.ndim == 2
        return theano.Apply(self, [y, l], [T.fvector()])

    def perform(self, node, inputs, outputs):
        y, l = inputs
        l = l.astype(np.int32)

        outputs[0][0] = ctc.loss_only(y, l)

    def grad(self, inputs, output_grads):
        y, l = inputs
        return [theano.gradient.grad_not_implemented(y, 0, ''),
                theano.gradient.grad_not_implemented(l, 1, '')]


ctc_equals = CTCEqualsOp()
ctc_cer = CTCCharacterErrorRate()
ctc_loss_op = CTCLossOpCpu()
ctc_loss_only = CTCLossOnlyOpCPU()


def ctc_loss(y, l):
    return ctc_loss_op(y, l)[0]
