import sys
from cffi import FFI
ffi = FFI()

with open('src/ctc.cpp') as f:
    extra_compile_flags = ['-std=c++1y']
    if sys.platform in ('linux', 'linux2'):
        extra_compile_flags.append('-fopenmp')

    ffi.set_source('ctc._libctc', f.read(), extra_compile_args=extra_compile_flags)

ffi.cdef('''
void ctc(
    const float* const y,
    const unsigned* const l,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const costs,
    float* const grad);

unsigned decode(
    unsigned* y,
    const unsigned timesteps,
    const unsigned alphabet_size,
    unsigned* out);

void equals(
    const unsigned* const y_pred,
    const unsigned* const y_true,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    unsigned char* const out);

''')


if __name__ == '__main__':
    ffi.compile()
