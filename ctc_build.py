import sys
from cffi import FFI
ffi = FFI()

with open('src/ctc.cpp') as f:
    extra_compile_args = ['-std=c++1y']
    extra_link_args = []
    if sys.platform in ('linux', 'linux2'):
        extra_compile_args.append('-fopenmp')
        extra_link_args = ['-lstdc++', '-lgomp']

    ffi.set_source('ctc._libctc',
                   f.read(),
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args,
                   source_extension='.cpp')

ffi.cdef('''
void ctc(
    const float* const y,
    const unsigned* const l,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const losses,
    float* const grad);

void ctc_loss_only(
    const float* const y,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const losses);

unsigned decode(
    const float* const y,
    const unsigned timesteps,
    const unsigned alphabet_size,
    unsigned* const out);

void equals(
    const float* const y_pred,
    const unsigned* const y_true,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    unsigned char* const out);

void edit_distance(
    const float* const y_pred,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const out);

void character_error_rate(
    const float* const y_pred,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const out);

''')


if __name__ == '__main__':
    ffi.compile()
