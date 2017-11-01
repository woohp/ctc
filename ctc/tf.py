from __future__ import absolute_import
import os
import glob
import tensorflow as tf
_ctc = tf.load_op_library(glob.glob(os.path.dirname(__file__) + '/_tf_op*.so')[0])


ctc_equals = _ctc.ctc_equals
ctc_cer = _ctc.ctc_cer
