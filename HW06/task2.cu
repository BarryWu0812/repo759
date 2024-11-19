#include <cuda.h>
#include <iostream>
#include <random>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    float* h_image = new float[n];
    float* h_mask = new float[2 * R + 1];
    float* h_output = new float[n];

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (unsigned int i = 0; i < n; i++)
    {
        h_image[i] = dist(generator);
        // h_image[i] = i;
    }

    for (unsigned int i = 0; i < 2 * R + 1; i++)
    {
        h_mask[i] = dist(generator);
        // h_mask[i] = 1;
    }

    float *d_image, *d_mask, *d_output;
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << h_output[n - 1] << std::endl;
    std::cout << elapsed_time << std::endl;

 
    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    return 0;
}
