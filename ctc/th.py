from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T

import ctc


class CTCEqualsOp(theano.Op):
    def make_node(self, y_pred, y_true):
        y_pred = T.as_tensor_variable(y_pred)
        y_true = T.as_tensor_variable(y_true)
        assert y_pred.ndim == 3
        assert y_true.ndim == 2
        return theano.Apply(self, [y_pred, y_true], [T.bvector()])

    def perform(self, node, inputs, outputs):
        y_pred, y_true = inputs
        y_true = y_true.astype(np.uint32)

        outputs[0][0] = ctc.equals(y_pred, y_true).astype(np.int8)


ctc_equals = CTCEqualsOp()


class CTCLossMixin(object):
    def grad(self, inputs, output_grads):
        y, l = inputs
        grad = self(y, l)[1]
        return [grad * output_grads[0][:, None, None],
                theano.gradient.grad_not_implemented(l, 1, '')]


class CTCLossOpCpu(theano.Op, CTCLossMixin):
    def make_node(self, y, l):
        y = T.as_tensor_variable(y)
        l = T.as_tensor_variable(l)
        assert y.ndim == 3
        assert l.ndim == 2
        return theano.Apply(self, [y, l], [T.fvector(), T.ftensor3()])

    def perform(self, node, inputs, outputs):
        y, l = inputs
        l = l.astype(np.uint32)

        outputs[0][0], outputs[1][0] = ctc.ctc(y, l)


class CTCLossOnlyOpCPU(theano.Op):
    def make_node(self, y, l):
        y = T.as_tensor_variable(y)
        l = T.as_tensor_variable(l)
        assert y.ndim == 3
        assert l.ndim == 2
        return theano.Apply(self, [y, l], [T.fvector()])

    def perform(self, node, inputs, outputs):
        y, l = inputs
        l = l.astype(np.uint32)

        outputs[0][0] = ctc.ctc_loss_only(y, l)

    def grad(self, inputs, output_grads):
        y, l = inputs
        return [theano.gradient.grad_not_implemented(y, 0, ''),
                theano.gradient.grad_not_implemented(l, 1, '')]


ctc_loss_op = CTCLossOpCpu()
ctc_loss_only_op = CTCLossOnlyOpCPU()


def ctc_loss(y, l):
    return ctc_loss_op(y, l)[0]


def ctc_loss_only(y, l):
    return ctc_loss_only_op(y, l)[0]
