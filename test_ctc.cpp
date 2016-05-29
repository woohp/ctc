#include <iostream>
#include "ctc.h"
using namespace std;


int main()
{
    unsigned l[] = {0};
    float y[] = {
        0, 0, 0, 30,
        30, 0, 0, 0,
        0, 0, 0, 30
    };
    float grad[sizeof(y) / sizeof(y[0])];
    float costs[1];
    int batch_size = 1;

    auto start = std::chrono::high_resolution_clock::now();
    ctc(y, l, batch_size, sizeof(y) / sizeof(y[0]) / 4 / batch_size, 4, sizeof(l) / sizeof(l[0]) / batch_size, costs, grad);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    printf("cost: %f %f\n", costs[0]);
    std::cout << "elapsed: " << diff.count() << " s\n";

    return 0;
}
