#include <cstring>
#include "utils.hpp"
using namespace std;


extern "C" {

unsigned decode(const float* __restrict__ const y,
                const unsigned timesteps,
                const unsigned alphabet_size,
                unsigned* __restrict__ const out)
{
    const auto blank = alphabet_size - 1;
    auto prev = blank;
    unsigned decoded_length = 0;

    for (unsigned t = 0; t < timesteps; t++)
    {
        auto y_t = y + t * alphabet_size;

        auto max = y_t + blank;
        for (auto it = max - 1; it >= y_t; it--)
        {
            if (*it > *max)
                max = it;
        }
        auto k = max - y_t;

        if (k != blank and k != prev)
        {
            out[decoded_length++] = k;
            prev = k;
        }
        else if (k == blank)
            prev = blank;
    }

    return decoded_length;
}

void equals(const float* __restrict__ const y_pred,
            const unsigned* __restrict__ const labels,
            const unsigned batches,
            const unsigned timesteps,
            const unsigned alphabet_size,
            const unsigned* label_lengths,
            unsigned char* __restrict__ const out)
{
    const auto blank = alphabet_size - 1;
    const auto y_size = timesteps * alphabet_size;
    const auto max_label_length = *max_element(label_lengths, label_lengths + batches);

    parallel_for(batches, 64, [=](unsigned batch)
    {
        auto const batch_y_pred = y_pred + batch * y_size;
        auto const batch_labels = labels + batch * max_label_length;

        bool equal = true;
        auto prev = blank;
        unsigned decoded_length = 0;

        for (unsigned t = 0; t < timesteps; t++)
        {
            auto y_t = batch_y_pred + t * alphabet_size;

            // argmax along the alphabets axis
            auto max = y_t + blank;
            for (auto it = max - 1; it >= y_t; it--)
            {
                if (*it > *max)
                    max = it;
            }
            auto k = max - y_t;

            if (k != blank and k != prev)
            {
                if (k != batch_labels[decoded_length++])
                {
                    equal = false;
                    break;
                }
                prev = k;
            }
            else if (k == blank)
                prev = blank;
        }

        equal = equal && (decoded_length == label_lengths[batch]);

        out[batch] = equal;
    });
}

void edit_distance(const float* __restrict__ const y_pred,
                   const unsigned* __restrict__ const labels,
                   const unsigned batches,
                   const unsigned timesteps,
                   const unsigned alphabet_size,
                   const unsigned* label_lengths,
                   float* __restrict__ const out)
{
    const auto blank = alphabet_size - 1;
    const auto y_size = timesteps * alphabet_size;
    const auto max_label_length = *max_element(label_lengths, label_lengths + batches);
    auto* __restrict__ const workspace = new float[max_label_length * 2 * batches];

    parallel_for(batches, 64, [=](unsigned batch)
    {
        auto const batch_y_pred = y_pred + batch * y_size;
        auto const batch_labels = labels + batch * max_label_length;

        auto* __restrict__ const batch_workspace = workspace + batch * max_label_length * 2;
        const auto batch_label_length = label_lengths[batch];
        for (unsigned s = 0; s < batch_label_length; s++)
            batch_workspace[s] = s + 1;

        auto prev = blank;
        unsigned decoded_length = 0;

        for (unsigned t = 0; t < timesteps; t++)
        {
            auto y_t = batch_y_pred + t * alphabet_size;

            // argmax along the alphabets axis
            auto max = y_t + blank;
            for (auto it = max - 1; it >= y_t; it--)
            {
                if (*it > *max)
                    max = it;
            }
            auto k = max - y_t;

            if (k != blank and k != prev)
            {
                auto* const d_t = batch_workspace + ((decoded_length + 1) % 2) * batch_label_length;
                auto* const d_tm1 = batch_workspace + (decoded_length % 2) * batch_label_length;

                if (k == batch_labels[0])
                    d_t[0] = decoded_length;
                else
                    d_t[0] = min<float>(min<float>(d_tm1[0] + 1, decoded_length + 2), decoded_length + 1);

                for (unsigned s = 1; s < batch_label_length; s++)
                {
                    if (k == batch_labels[s])
                        d_t[s] = d_tm1[s-1];
                    else
                        d_t[s] = min(min(d_tm1[s] + 1, d_tm1[s-1] + 1), d_t[s-1] + 1);
                }

                decoded_length++;
                prev = k;
            }
            else if (k == blank)
                prev = blank;
        }

        if (decoded_length > 0)
        {
            const auto d_t_last = batch_workspace + (decoded_length % 2) * batch_label_length;
            out[batch] = d_t_last[batch_label_length - 1];
        }
        else
            out[batch] = batch_label_length;
    });

    delete[] workspace;
}

void character_error_rate(const float* __restrict__ const y_pred,
                          const unsigned* __restrict__ const labels,
                          const unsigned batches,
                          const unsigned timesteps,
                          const unsigned alphabet_size,
                          const unsigned* label_lengths,
                          float* __restrict__ const out)
{
    edit_distance(y_pred, labels, batches, timesteps, alphabet_size, label_lengths, out);
    for (unsigned batch = 0; batch < batches; batch++)
        out[batch] /= label_lengths[batch];
}


}
