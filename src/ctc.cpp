#include <cmath>
#include <array>
#include <cstring>
#include <algorithm>
#include <future>
#include <thread>
using namespace std;


constexpr float ALMOST_ZERO = 2e-9f;
constexpr float LOG_ALMOST_ZERO = -20.0f;

inline float safe_log(float y)
{
    if (y < ALMOST_ZERO)
        return LOG_ALMOST_ZERO;
    return log(y);
};


float logadd(float x, float y)
{
    if (x == y)
        return x + M_LN2;

    const float diff = x - y;
    if (diff > 0)
        return x + log1p(exp(-diff));
    else
        return y + log1p(exp(diff));
}


float logadd(float x, float y, float z)
{
    if (x >= y and x >= z)  // x is largest
        return x + log1p(exp(y - x) + exp(z - x));
    else if (y >= x and y >= z) // y is largest
        return y + log1p(exp(x - y) + exp(z - y));
    else  // z is largest
        return z + log1p(exp(x - z) + exp(y - z));
}


float logadd(float* x, unsigned n)
{
    if (n == 1)
        return x[0];
    else if (n == 2)
        return logadd(x[0], x[1]);
    else if (n == 3)
        return logadd(x[0], x[1], x[2]);
    const float max = *max_element(x, x+n);
    float sum = -1.0f;
    for (unsigned i = 0; i < n; i++)
        sum += exp(x[i] - max);
    return max + log1p(sum);
}


template<typename F>
void parallel_for(unsigned length, unsigned max_threads, const F& func)
{
    const auto n_threads = min(std::thread::hardware_concurrency(), max_threads);
    const auto thread_stride = length / n_threads;
    auto leftover = length - thread_stride * n_threads;

    auto func_wrapper = [&func](unsigned start, unsigned end)
    {
        for (; start < end; start++)
            func(start);
    };

    array<future<void>, 32> futures;
    for (unsigned i = 0, start = 0; i < n_threads; i++)
    {
        auto this_thread_stride = thread_stride;
        if (leftover > 0)
        {
            this_thread_stride += 1;
            leftover--;
        }
        futures[i] = std::async(std::launch::async, func_wrapper, start, start + this_thread_stride);
        start += this_thread_stride;
    }
    for (unsigned i = 0; i < n_threads; i++)
        futures[i].wait();
}


extern "C" {

void ctc(const float* __restrict__ const y,
         const unsigned* __restrict__ const labels,
         const unsigned batches,
         const unsigned timesteps,
         const unsigned alphabet_size,
         const unsigned labels_length,
         float* __restrict__ const losses,
         float* __restrict__ const grad)
{
    const auto labels_length_p = labels_length * 2 + 1;
    const auto y_size = timesteps * alphabet_size;
    const auto a_size = timesteps * labels_length_p;
    const auto b_size = a_size;
    const auto alphabet_counts_size = alphabet_size - 1;  // not counting blank
    const auto alphabet_indices_size = (alphabet_size - 1) * labels_length;
    const auto workspace_size = y_size + a_size + b_size + labels_length_p + alphabet_indices_size + alphabet_counts_size;

    const auto blank = alphabet_size - 1;

    static_assert(sizeof(float) == sizeof(unsigned), "wrong size");

    auto* __restrict__ const workspace = new float[workspace_size * batches];

    parallel_for(batches, 64, [=](unsigned batch)
    {
        const auto* __restrict__ const batch_y = y + batch * y_size;
        const auto* __restrict__ const batch_labels = labels + batch * labels_length;
        auto* __restrict__ const batch_grad = grad + batch * y_size;

        auto* __restrict__ const batch_workspace = workspace + batch * workspace_size;
        auto* __restrict__ const log_y = batch_workspace;
        auto* __restrict__ const a = log_y + y_size;
        auto* __restrict__ const b = a + a_size;
        auto* __restrict__ const numbers_to_add = b + b_size;
        auto* __restrict__ const alphabet_indices = reinterpret_cast<unsigned*>(numbers_to_add + labels_length_p);
        auto* __restrict__ const alphabet_counts = alphabet_indices + alphabet_indices_size;

        for (unsigned i = 0; i < y_size; i++)
            log_y[i] = safe_log(batch_y[i]);

        // forward step
        a[0] = log_y[blank];
        a[1] = log_y[batch_labels[0]];
        for (unsigned s = 2; s < labels_length_p; s++)
            a[s] = LOG_ALMOST_ZERO;

        for (unsigned t = 1; t < timesteps; t++)
        {
            auto* const a_t = a + t * labels_length_p;
            const auto* const a_tm1 = a_t - labels_length_p;
            const auto* const log_y_t = log_y + t * alphabet_size;

            unsigned s = 0;
            a_t[s] = log_y_t[blank] + a_tm1[s];

            s = 1;
            a_t[s] = log_y_t[batch_labels[0]] + logadd(a_tm1[s], a_tm1[s-1]);

            s = 2;
            a_t[s] = log_y_t[blank] + logadd(a_tm1[s], a_tm1[s-1]);

            for (unsigned _s = 1; _s < labels_length; _s++)
            {
                s = 2 * _s + 1;
                if (batch_labels[_s-1] == batch_labels[_s])
                    a_t[s] = log_y_t[batch_labels[_s]] + logadd(a_tm1[s], a_tm1[s-1]);
                else
                    a_t[s] = log_y_t[batch_labels[_s]] + logadd(a_tm1[s], a_tm1[s-1], a_tm1[s-2]);

                s++;
                a_t[s] = log_y_t[blank] + logadd(a_tm1[s], a_tm1[s-1]);
            }
        }

        // backward step
        auto* const b_tl = b + (timesteps - 1) * labels_length_p;
        b_tl[labels_length_p-1] = log_y[(timesteps-1)*alphabet_size + blank];
        b_tl[labels_length_p-2] = log_y[(timesteps-1)*alphabet_size + batch_labels[labels_length-1]];
        for (unsigned s = 0; s < labels_length_p - 2; s++)
            b_tl[s] = LOG_ALMOST_ZERO;

        for (unsigned t = timesteps - 2; t < timesteps; t--) // t < timesteps because it will wrap-around
        {
            auto* b_t = b + t * labels_length_p;
            auto* b_tp1 = b_t + labels_length_p;
            auto* log_y_t = log_y + t * alphabet_size;

            auto s = labels_length_p - 1;
            b_t[s] = log_y_t[blank] + b_tp1[s];

            s = labels_length_p - 2;
            b_t[s] = log_y_t[batch_labels[labels_length-1]] + logadd(b_tp1[s], b_tp1[s+1]);

            s = labels_length_p - 3;
            b_t[s] = log_y_t[blank] + logadd(b_tp1[s], b_tp1[s+1]);

            for (unsigned _s = labels_length - 2; _s < labels_length; _s--) // _s < labels_length because it will wrap-around
            {
                s = 2 * _s + 1;
                if (batch_labels[_s+1] == batch_labels[_s])
                    b_t[s] = log_y_t[batch_labels[_s]] + logadd(b_tp1[s], b_tp1[s+1]);
                else
                    b_t[s] = log_y_t[batch_labels[_s]] + logadd(b_tp1[s], b_tp1[s+1], b_tp1[s+2]);

                s--;
                b_t[s] = log_y_t[blank] + logadd(b_tp1[s], b_tp1[s+1]);
            }
        }

        const auto log_prob = logadd(a[timesteps*labels_length_p-1], a[timesteps*labels_length_p-2]);
        losses[batch] = -log_prob;

        // for each "letter", figure out all the indices in the label where that letter is present
        memset(alphabet_counts, 0, sizeof(alphabet_counts[0]) * alphabet_counts_size);
        for (unsigned _s = 0; _s < labels_length; _s++)
        {
            auto k = batch_labels[_s];
            alphabet_indices[k * labels_length + alphabet_counts[k]++] = _s * 2 + 1;
        }

        // calculate gradient
        for (unsigned t = 0; t < timesteps; t++)
        {
            auto* const a_t = a + t * labels_length_p;
            const auto* const b_t = b + t * labels_length_p;
            const auto* const log_y_t = log_y + t * alphabet_size;
            auto* const grad_t = batch_grad + t * alphabet_size;

            numbers_to_add[0] = (a_t[0] += b_t[0]) - log_y_t[blank];
            for (unsigned _s = 0; _s < labels_length; _s++)
            {
                unsigned s = _s * 2 + 1;
                numbers_to_add[s] = (a_t[s] += b_t[s]) - log_y_t[batch_labels[_s]];
                s++;
                numbers_to_add[s] = (a_t[s] += b_t[s]) - log_y_t[blank];
            }
            const auto log_prob_t = logadd(numbers_to_add, labels_length_p);

            // for non-blank labels
            for (unsigned k = 0; k < alphabet_size - 1; k++)
            {
                unsigned i;
                for (i = 0; i < alphabet_counts[k]; i++)
                {
                    unsigned s = alphabet_indices[k * labels_length + i];
                    numbers_to_add[i] = a_t[s];
                }
                if (i == 0)
                {
                    grad_t[k] = 0.0f;
                }
                else
                {
                    const auto grad_loss = logadd(numbers_to_add, i) - 2 * log_y_t[k] - log_prob_t;
                    grad_t[k] = -exp(grad_loss);
                }
            }

            // for blank
            unsigned i = 0;
            for (unsigned s = 0; s < labels_length_p; s += 2)
                numbers_to_add[i++] = a_t[s];
            auto grad_loss = logadd(numbers_to_add, i) - 2 * log_y_t[blank] - log_prob_t;
            grad_t[blank] = -exp(grad_loss);
        }
    });

    delete[] workspace;
}


void ctc_loss_only(
    const float* __restrict__ const y,
    const unsigned* __restrict__ const labels,
    const unsigned batches,
    const unsigned timesteps,
    const unsigned alphabet_size,
    const unsigned labels_length,
    float* __restrict__ const losses)
{
    const auto labels_length_p = labels_length * 2 + 1;
    const auto y_size = timesteps * alphabet_size;
    const auto log_y_size = alphabet_size;
    const auto a_size = 2 * labels_length_p;
    const auto workspace_size = log_y_size + a_size;

    const auto blank = alphabet_size - 1;

    static_assert(sizeof(float) == sizeof(unsigned), "wrong size");

    auto* __restrict__ const workspace = new float[workspace_size * batches];

    parallel_for(batches, 64, [=](unsigned batch)
    {
        const auto* __restrict__ const batch_y = y + batch * y_size;
        const auto* __restrict__ const batch_labels = labels + batch * labels_length;

        auto* __restrict__ const batch_workspace = workspace + batch * workspace_size;
        auto* __restrict__ const log_y = batch_workspace;
        auto* __restrict__ const a = log_y + log_y_size;

        // forward step
        a[0] = safe_log(batch_y[blank]);
        a[1] = safe_log(batch_y[batch_labels[0]]);
        for (unsigned s = 2; s < labels_length_p; s++)
            a[s] = LOG_ALMOST_ZERO;

        for (unsigned t = 1; t < timesteps; t++)
        {
            auto* const a_t = a + (t % 2) * labels_length_p;
            const auto* const a_tm1 = a + ((t - 1) % 2) * labels_length_p;
            const auto* const y_t = batch_y + t * alphabet_size;

            for (unsigned k = 0; k < alphabet_size; k++)
                log_y[k] = safe_log(y_t[k]);

            unsigned s = 0;
            a_t[s] = log_y[blank] + a_tm1[s];

            s = 1;
            a_t[s] = log_y[batch_labels[0]] + logadd(a_tm1[s], a_tm1[s-1]);

            s = 2;
            a_t[s] = log_y[blank] + logadd(a_tm1[s], a_tm1[s-1]);

            for (unsigned _s = 1; _s < labels_length; _s++)
            {
                s = 2 * _s + 1;
                if (batch_labels[_s-1] == batch_labels[_s])
                    a_t[s] = log_y[batch_labels[_s]] + logadd(a_tm1[s], a_tm1[s-1]);
                else
                    a_t[s] = log_y[batch_labels[_s]] + logadd(a_tm1[s], a_tm1[s-1], a_tm1[s-2]);

                s++;
                a_t[s] = log_y[blank] + logadd(a_tm1[s], a_tm1[s-1]);
            }
        }

        const auto a_t_last = a + ((timesteps - 1) % 2) * labels_length_p;
        const auto log_prob = logadd(a_t_last[labels_length_p-1], a_t_last[labels_length_p-2]);
        losses[batch] = -log_prob;
    });

    delete[] workspace;
}


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
            {
                max = it;
                if (*it > 0.5f) // short circuit if possible
                    break;
            }
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
            const unsigned labels_length,
            unsigned char* __restrict__ const out)
{
    const auto blank = alphabet_size - 1;
    const auto y_size = timesteps * alphabet_size;

    parallel_for(batches, 64, [=](unsigned batch)
    {
        auto const batch_y_pred = y_pred + batch * y_size;
        auto const batch_labels = labels + batch * labels_length;

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
                {
                    max = it;
                    if (*it > 0.5f) // short circuit if possible
                        break;
                }
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

        equal = equal && (decoded_length == labels_length);

        out[batch] = equal;
    });
}

void edit_distance(const float* __restrict__ const y_pred,
                   const unsigned* __restrict__ const labels,
                   const unsigned batches,
                   const unsigned timesteps,
                   const unsigned alphabet_size,
                   const unsigned labels_length,
                   float* __restrict__ const out)
{
    const auto blank = alphabet_size - 1;
    const auto y_size = timesteps * alphabet_size;
    auto* __restrict__ const workspace = new float[labels_length * 2 * batches];

    parallel_for(batches, 64, [=](unsigned batch)
    {
        auto const batch_y_pred = y_pred + batch * y_size;
        auto const batch_labels = labels + batch * labels_length;

        auto* __restrict__ const batch_workspace = workspace + batch * labels_length * 2;
        for (unsigned s = 0; s < labels_length; s++)
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
                {
                    max = it;
                    if (*it > 0.5f) // short circuit if possible
                        break;
                }
            }
            auto k = max - y_t;

            if (k != blank and k != prev)
            {
                auto* const d_t = batch_workspace + ((decoded_length + 1) % 2) * labels_length;
                auto* const d_tm1 = batch_workspace + (decoded_length % 2) * labels_length;

                if (k == batch_labels[0])
                    d_t[0] = decoded_length;
                else
                    d_t[0] = min<float>(min<float>(d_tm1[0] + 1, decoded_length + 2), decoded_length + 1);

                for (unsigned s = 1; s < labels_length; s++)
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
            const auto d_t_last = batch_workspace + (decoded_length % 2) * labels_length;
            out[batch] = d_t_last[labels_length - 1];
        }
        else
            out[batch] = labels_length;
    });

    delete[] workspace;
}

void character_error_rate(const float* __restrict__ const y_pred,
                          const unsigned* __restrict__ const labels,
                          const unsigned batches,
                          const unsigned timesteps,
                          const unsigned alphabet_size,
                          const unsigned labels_length,
                          float* __restrict__ const out)
{
    edit_distance(y_pred, labels, batches, timesteps, alphabet_size, labels_length, out);
    float denom = 1.0f / labels_length;
    for (unsigned batch = 0; batch < batches; batch++)
        out[batch] *= denom;
}


}
