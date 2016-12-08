from __future__ import absolute_import, print_function
from ._libctc import ffi, lib
import numpy as np


def loss(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.int32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    losses = np.empty(batches, dtype=np.float32)
    gradients = np.empty_like(y)

    lib.ctc(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('float*', losses.ctypes.data),
        ffi.cast('float*', gradients.ctypes.data)
    )

    return losses, gradients


def loss_only(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.int32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    losses = np.empty(batches, dtype=np.float32)

    lib.ctc_loss_only(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('float*', losses.ctypes.data),
    )

    return losses


def decode(y):
    assert len(y.shape) == 2
    assert y.dtype == np.float32

    timesteps, alphabet_size = y.shape
    decoded = np.empty(len(y), dtype=np.int32)
    decoded_length = lib.decode(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned*', decoded.ctypes.data)
    )

    return decoded[:decoded_length]


def equals(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.int32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    out = np.empty(len(y), dtype=np.bool)

    lib.equals(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('unsigned char*', out.ctypes.data),
    )

    return out


def edit_distance(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.int32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    out = np.empty(len(y), dtype=np.float32)

    lib.edit_distance(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('float*', out.ctypes.data),
    )

    return out


def character_error_rate(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.int32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    out = np.empty(len(y), dtype=np.float32)

    lib.character_error_rate(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('float*', out.ctypes.data),
    )

    return out


if __name__ == '__main__':
    y = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.01, 0.01, 0.97, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 0.01],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.01, 0.01, 0.97, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97],
         [0.97, 0.97, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97]] * 15
    l = [0, 1, 2, 1, 0] * 15

    y = np.array([y, y], dtype=np.float32)
    l = np.array([l, l], dtype=np.int32)

    import time

    start = time.time()
    losses, gradients = loss(y, l)
    print(time.time() - start)
    print(losses)
