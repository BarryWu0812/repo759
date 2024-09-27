#include "scan.h"
#include <cstdio>

void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0)
        return;

    float tmp = 0;
    
    for (std::size_t i = 0; i < n; i++)
    {
        tmp += arr[i];
        output[i] = tmp;
        // printf("output[%lu]: %f\n", i, output[i]);
    }
}