# ctc

An implementation of the Connectionist Temporal Classification loss function for CPU, with bindings for Theano and Tensorflow.

## Introduction

[Connectionist Temporal Classification](http://www.cs.toronto.edu/~graves/icml_2006.pdf) is a loss function for doing supervised sequence to sequence learning without explicit segmentation.
This is an implementation of the loss function for the CPU.

## Examples
### Simple example
```python
import numpy as np
import ctc

y_pred = np.array([
    [[0.01, 0.01, 0.01, 0.97],
     [0.01, 0.01, 0.97, 0.01],
     [0.01, 0.97, 0.01, 0.01],
     [0.01, 0.01, 0.01, 0.97]],
    [[0.01, 0.01, 0.01, 0.97],
     [0.01, 0.97, 0.01, 0.01],
     [0.01, 0.97, 0.01, 0.01],
     [0.01, 0.01, 0.01, 0.97]]
], dtype=np.float32)
labels = np.array([[2, 1], [1, 2]], dtype=np.int32)

loss, gradient = ctc.loss(y_pred, labels)
print(loss)
print(gradient)

# prints:
[ 0.10090086  3.97769332]
[[[  0.00000000e+00  -2.21158181e-09  -1.04133022e+00  -1.02019238e+00]
  [  0.00000000e+00  -1.06225219e-02  -1.03070879e+00  -1.06236162e-02]
  [  0.00000000e+00  -1.03070867e+00  -1.06225219e-02  -1.06236162e-02]
  [  0.00000000e+00  -1.04133022e+00  -2.21158181e-09  -1.02019238e+00]]

 [[  0.00000000e+00  -1.03081965e+00  -3.21353921e-09  -1.02030087e+00]
  [  0.00000000e+00  -1.02558684e+00  -1.04117803e-02  -5.07667661e-01]
  [  0.00000000e+00  -5.12740016e-01  -4.97514267e+01  -5.12792826e-01]
  [  0.00000000e+00  -1.02523682e-05  -5.07510643e+01  -5.07720709e-01]]]
```

### Theano integration
```python
import theano
import theano.tensor as T
import ctc.th

y_pred_ = T.ftensor3()
labels_ = T.imatrix()
loss = ctc.th.ctc_loss(y_pred_, labels_)
loss_function = theano.function([y_pred_, labels_], loss)
grad_function = theano.function([y_pred_, labels_], T.grad(loss.sum(), y_pred_))

print(loss_function(y_pred, labels))
print(grad_function(y_pred, labels))
```

### Tensorflow integration
```python
import tensorflow as tf
import ctc.tf

y_pred_ = tf.placeholder(name='y', shape=(None, None, None), dtype=tf.float32)
labels_ = tf.placeholder(name='l', shape=(None, None), dtype=tf.int32)
loss = ctc.tf.ctc_loss(y_pred_, labels_)
grad = tf.gradients([loss], [y_pred_])
with tf.Session('') as session:
    print(session.run(loss, {y_pred_: y_pred, labels_: labels}))
    print(session.run(grad, {y_pred_: y_pred, labels_: labels}))
```

## Notes
* The loss operates on the probabilities (post softmax), not the raw output output of the final layer. This is similar to most other CTC implementations, but unlike Baidu's warp-ctc.
* The blank symbol is the final symbol.
* All inputs must have the same shape, and all labels must have the same length. This makes it less flexible than some other implementations, but makes it simpler to use.
* The implementation calculates the loss and gradient in logspace. Care is taken to ensure that it does not blow up when receiving 0's.
