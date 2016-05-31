from _libctc import ffi, lib
import numpy as np


def ctc(y, labels):
    assert len(y.shape) == 3
    assert len(labels.shape) == 2
    assert len(y) == len(labels)
    assert y.dtype == np.float32
    assert labels.dtype == np.uint32

    batches, timesteps, alphabet_size = y.shape
    labels_length = labels.shape[1]
    costs = np.empty(batches, dtype=np.float32)
    gradients = np.empty_like(y)

    lib.ctc(
        ffi.cast('float*', y.ctypes.data),
        ffi.cast('unsigned*', labels.ctypes.data),
        ffi.cast('unsigned', batches),
        ffi.cast('unsigned', timesteps),
        ffi.cast('unsigned', alphabet_size),
        ffi.cast('unsigned', labels_length),
        ffi.cast('float*', costs.ctypes.data),
        ffi.cast('float*', gradients.ctypes.data)
    )

    return costs, gradients


def decode(y):
    assert len(y.shape) == 2
    assert y.dtype == np.float32

    timesteps, alphabet_size = y.shape
    decoded = np.empty(len(y), dtype=np.uint32)
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
    assert labels.dtype == np.uint32

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
    l = np.array([l, l], dtype=np.uint32)

    import time

    start = time.time()
    costs, gradients = ctc(y, l)
    print time.time() - start
    print costs
