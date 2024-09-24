#include "scan.h"
#include <cstdio>

void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) return;
    
    for (std::size_t i = 0; i < n; i++) {
        output[i] = arr[i];
        printf("output[%lu]: %f\n", i, output[i]);
    }
}

int main() {
    const std::size_t n = 5;
    float arr[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[n];
    
    scan(arr, output, n);
    
    return 0;
}