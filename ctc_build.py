from cffi import FFI
ffi = FFI()

ffi.set_source('_ctc', '''
void ctc(const float* const y,
         const unsigned* const l,
         const size_t batches,
         const size_t timesteps,
         const size_t alphabet_size,
         const size_t labels_length,
         float* const costs,
         float* const grad);
''')


if __name__ == '__main__':
    ffi.compile()
