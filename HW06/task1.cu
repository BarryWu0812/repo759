#include <cuda.h>
#include <iostream>
#include <random>
#include "matmul.cuh"

int main(int argc, char* argv[])
{
    size_t n = std::atoi(argv[1]);
    unsigned int threadsPerBlock = std::atoi(argv[2]);
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1., 1.);
    float* h_a = new float[n * n];
    float* h_b = new float[n * n];
    float* h_c = new float[n * n];
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    for (unsigned int i = 0; i < n * n; i++)
    {
        h_a[i] = dist(generator);
        h_b[i] = dist(generator);
        // h_a[i] = i;
        // h_b[i] = i;
        // std::cout << h_a[i] * h_b[i] << std::endl;
    }
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * n * n);
    cudaMalloc((void**)&d_b, sizeof(float) * n * n);
    cudaMalloc((void**)&d_c, sizeof(float) * n * n);
    cudaMemcpy(d_a, h_a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(startEvent, 0);
    matmul(d_a, d_b, d_c, n, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(h_c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    // std::cout << d_c[n * n - 1] << std::endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << h_c[n * n - 1] << std::endl;
    std::cout << elapsedTime << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
