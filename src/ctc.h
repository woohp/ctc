extern "C"
{

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
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    unsigned char* const out);

void character_error_rate(
    const float* const y_pred,
    const unsigned* const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* const out);

}
