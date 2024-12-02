#include <cuda.h>
#include <iostream>
#include "reduce.cuh"
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add on load from global memory to shared memory
    sdata[tid] = (i < n ? g_idata[i] : 0) + (i + blockDim.x < n ? g_idata[i + blockDim.x] : 0);
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x; s / 2 > 0; s >>= 1) {
        if (tid < s / 2)
        {
            // sdata[tid] += (tid < s - 1 ? sdata[tid + s + 1] : 0);
            sdata[tid] += sdata[tid + s / 2];
            if (s % 2 == 1 && tid == s / 2 - 1)
            {
                sdata[tid] += sdata[tid + s / 2 + 1];
            }
        }
        __syncthreads();
    }

    // Write the result of this block's reduction to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block)
{
    unsigned int blocks_per_grid = (N + (threads_per_block * 2 - 1)) / (threads_per_block * 2);

    // Configure shared memory size
    size_t shared_mem_size = threads_per_block * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Perform reduction repeatedly until only one element is left
    while (blocks_per_grid > 1)
    {
        reduce_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // Update input and output for the next iteration
        float *temp = *input;
        *input = *output;
        *output = temp;

        N = blocks_per_grid;
        blocks_per_grid = (N + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }

    // Final reduction for the last block
    reduce_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(*input, *output, N);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Retrieve the result
    float result;
    cudaMemcpy(&result, *output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}