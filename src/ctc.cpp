#include <cmath>
#include <cstring>
#include <algorithm>
using namespace std;


#define ALMOST_ZERO 2e-9f
#define LOG_ALMOST_ZERO -20.0f


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


extern "C" {

void ctc(const float* __restrict__ const y,
         const unsigned* __restrict__ const l,
         const unsigned batches,
         const unsigned timesteps,
         const unsigned alphabet_size,
         const unsigned labels_length,
         float* __restrict__ const costs,
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

    #pragma omp parallel for
    for (unsigned batch = 0; batch < batches; batch++)
    {
        const auto* __restrict__ const batch_y = y + batch * y_size;
        auto* __restrict__ const batch_grad = grad + batch * y_size;

        auto* __restrict__ const batch_workspace = workspace + batch * workspace_size;
        auto* __restrict__ const log_y = batch_workspace;
        auto* __restrict__ const a = log_y + y_size;
        auto* __restrict__ const b = a + a_size;
        auto* __restrict__ const numbers_to_add = b + b_size;
        auto* __restrict__ const alphabet_indices = reinterpret_cast<unsigned*>(numbers_to_add + labels_length_p);
        auto* __restrict__ const alphabet_counts = alphabet_indices + alphabet_indices_size;

        for (unsigned i = 0; i < y_size; i++)
        {
            if (batch_y[i] < ALMOST_ZERO)
                log_y[i] = LOG_ALMOST_ZERO;
            else
                log_y[i] = log(batch_y[i]);
        }

        // forward step
        a[0] = log_y[blank];
        a[1] = log_y[l[0]];
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
            a_t[s] = log_y_t[l[0]] + logadd(a_tm1[s], a_tm1[s-1]);

            s = 2;
            a_t[s] = log_y_t[blank] + logadd(a_tm1[s], a_tm1[s-1]);

            for (unsigned _s = 1; _s < labels_length; _s++)
            {
                s = 2 * _s + 1;
                if (l[_s-1] == l[_s])
                    a_t[s] = log_y_t[l[_s]] + logadd(a_tm1[s], a_tm1[s-1]);
                else
                    a_t[s] = log_y_t[l[_s]] + logadd(a_tm1[s], a_tm1[s-1], a_tm1[s-2]);

                s++;
                a_t[s] = log_y_t[blank] + logadd(a_tm1[s], a_tm1[s-1]);
            }
        }

        // backward step
        auto* const b_tl = b + (timesteps - 1) * labels_length_p;
        b_tl[labels_length_p-1] = log_y[(timesteps-1)*alphabet_size + blank];
        b_tl[labels_length_p-2] = log_y[(timesteps-1)*alphabet_size + l[labels_length-1]];
        for (unsigned s = 0; s < labels_length_p - 2; s++)
            b_tl[s] = LOG_ALMOST_ZERO;

        for (unsigned t = timesteps - 2; t < timesteps; t--) // t < timesteps because it will wraparound
        {
            auto* b_t = b + t * labels_length_p;
            auto* b_tp1 = b_t + labels_length_p;
            auto* log_y_t = log_y + t * alphabet_size;

            auto s = labels_length_p - 1;
            b_t[s] = log_y_t[blank] + b_tp1[s];

            s = labels_length_p - 2;
            b_t[s] = log_y_t[l[labels_length-1]] + logadd(b_tp1[s], b_tp1[s+1]);

            s = labels_length_p - 3;
            b_t[s] = log_y_t[blank] + logadd(b_tp1[s], b_tp1[s+1]);

            for (unsigned _s = labels_length - 2; _s < timesteps; _s--) // t < timesteps because it will wraparound
            {
                s = 2 * _s + 1;
                if (l[_s+1] == l[_s])
                    b_t[s] = log_y_t[l[_s]] + logadd(b_tp1[s], b_tp1[s+1]);
                else
                    b_t[s] = log_y_t[l[_s]] + logadd(b_tp1[s], b_tp1[s+1], b_tp1[s+2]);

                s--;
                b_t[s] = log_y_t[blank] + logadd(b_tp1[s], b_tp1[s+1]);
            }
        }

        const auto log_prob = logadd(a[timesteps*labels_length_p-1], a[timesteps*labels_length_p-2]);
        costs[batch] = -log_prob;

        // for each "letter", figure out all the indices in the label where that letter is present
        memset(alphabet_counts, 0, sizeof(alphabet_counts[0]) * alphabet_counts_size);
        for (unsigned _s = 0; _s < labels_length; _s++)
        {
            auto k = l[_s];
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
                numbers_to_add[s] = (a_t[s] += b_t[s]) - log_y_t[l[_s]];
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
                    const auto grad_cost = logadd(numbers_to_add, i) - 2 * log_y_t[k] - log_prob_t;
                    grad_t[k] = -exp(grad_cost);
                }
            }

            // for blank
            unsigned i = 0;
            for (unsigned s = 0; s < labels_length_p; s += 2)
                numbers_to_add[i++] = a_t[s];
            auto grad_cost = logadd(numbers_to_add, i) - 2 * log_y_t[blank] - log_prob_t;
            grad_t[blank] = -exp(grad_cost);
        }
    }

    delete[] workspace;
}

unsigned decode(const unsigned* __restrict__ const y,
                const unsigned timesteps,
                const unsigned alphabet_size,
                unsigned* __restrict__ const out)
{
    auto prev = alphabet_size;
    unsigned decoded_length = 0;

    for (unsigned i = 0; i < timesteps; i++)
    {
        if (y[i] != alphabet_size and y[i] != prev)
        {
            out[decoded_length++] = y[i];
            prev = y[i];
        }
        else if (y[i] == alphabet_size)
            prev = alphabet_size;
    }

    return decoded_length;
}

void equals(const unsigned* __restrict__ const y_pred,
            const unsigned* __restrict__ const y_true,
            const unsigned batches,
            const unsigned timesteps,
            const unsigned alphabet_size,
            const unsigned labels_length,
            unsigned char* __restrict__ const out)
{
    #pragma omp parallel for
    for (unsigned batch = 0; batch < batches; batch++)
    {
        auto batch_y_pred = y_pred + batch * timesteps;
        auto batch_y_true = y_true + batch * labels_length;

        bool equal = true;

        auto prev = alphabet_size;
        unsigned decoded_length = 0;
        for (unsigned i = 0; i < timesteps; i++)
        {
            if (batch_y_pred[i] != alphabet_size and batch_y_pred[i] != prev)
            {
                if (batch_y_pred[i] != batch_y_true[decoded_length++])
                {
                    equal = false;
                    break;
                }
                prev = batch_y_pred[i];
            }
            else if (batch_y_pred[i] == alphabet_size)
                prev = alphabet_size;
        }

        out[batch] = equal;
    }
}

}
