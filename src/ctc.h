extern "C"
{

unsigned decode(
    const float* const y,
    const unsigned timesteps,
    const unsigned alphabet_size,
    unsigned* const out);

void equals(
    const float* const y_pred,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned* label_lengths,
    unsigned char* const out);

void character_error_rate(
    const float* const y_pred,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned* label_lengths,
    float* const out);

}
