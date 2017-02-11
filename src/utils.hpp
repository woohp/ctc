#ifndef _UTILS_HPP
#define _UTILS_HPP
#include <algorithm>
#include <array>
#include <cmath>
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
    if (length == 1)
    {
        func(0);
        return;
    }

    const auto n_threads = min({std::thread::hardware_concurrency(), max_threads, length});
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


#endif
