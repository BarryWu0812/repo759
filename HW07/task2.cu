#include <cuda.h>
#include <iostream>
#include <random>
#include "reduce.cuh"

int main(int argc, char* argv[]) {
    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> distf(-1.0, 1.0);


    float *h_input = new float[n];
    for (unsigned int i = 0; i < n; i++)
    {
        h_input[i] = distf(generator);
        // h_input[i] = i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, ((n + threads_per_block * 2 - 1) / (threads_per_block * 2)) * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the reduce function
    reduce(&d_input, &d_output, n, threads_per_block);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);


    delete[] h_input;

    return 0;
}
