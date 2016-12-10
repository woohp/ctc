from __future__ import absolute_import
import os
import glob
import tensorflow as tf
from tensorflow.python.framework import ops
_ctc = tf.load_op_library(glob.glob(os.path.dirname(__file__) + '/_tf_op.*.so')[0])


@ops.RegisterGradient('CTC')
def _ctc_grad(op, output_grad1, output_grad2):
    grad = op.outputs[1]
    return [grad * output_grad1[:, tf.newaxis, tf.newaxis], None]


def ctc_loss(y, l):
    return _ctc.ctc(y, l).loss


ctc_loss_only = _ctc.ctc_loss_only
ctc_equals = _ctc.ctc_equals
ctc_cer = _ctc.ctc_cer
