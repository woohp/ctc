extern "C"
{

void ctc(
    const float* __restrict__ const y,
    const unsigned* __restrict__ const l,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* __restrict__ const costs,
    float* __restrict__ const grad);

unsigned decode(
    const unsigned* __restrict__ const y,
    const unsigned timesteps,
    const unsigned alphabet_size,
    unsigned* __restrict__ const out);

void equals(
    const unsigned* __restrict__ const y_pred,
    const unsigned* __restrict__ const y_true,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    unsigned char* __restrict__ const out);

}
