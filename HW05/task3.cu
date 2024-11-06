#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"

int main(int argc, char* argv[])
{
    unsigned int n = std::atoi(argv[1]);
    const int threadsPerBlock = 16;
    const int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dista(-10., 10.);
    std::uniform_real_distribution<float> distb(0., 1.);
    float* h_a = new float[n];
    float* h_b = new float[n];
    float *d_a, *d_b;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaMalloc((void**)&d_a, sizeof(float) * n);
    cudaMalloc((void**)&d_b, sizeof(float) * n);
    for (unsigned int i = 0; i < n; i++)
    {
        // image[i] = i;
        h_a[i] = dista(generator);
        h_b[i] = distb(generator);
        // std::cout << h_a[i] * h_b[i] << std::endl;
    }
    // std::cout << "=====" << std::endl;

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(startEvent, 0);
    //invoke GPU kernel, with one block that has four threads
    vscale<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, n);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(h_b, d_b, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // std::cout << "Values stored in hA: " << std::endl;
    // for (unsigned int i = 0; i < n; i++)
    //     std::cout << h_b[i] << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << elapsedTime << std::endl;
    std::cout << h_b[0] << std::endl;
    std::cout << h_b[n-1] << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
