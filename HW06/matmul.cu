#include <cuda.h>
#include <iostream>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / n;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) % n;

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            // printf("A: %f, B: %f\n", A[row * n + k], B[k * n + col]);
            sum += A[row * n + k] * B[k * n + col];
        }
        // printf("row: %d, col: %d\n", row, col);
        C[row * n + col] = sum;
        // printf("C: %f\n", C[row * n + col]);
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    unsigned int blocksPerGrid = (n * n + threads_per_block - 1) / threads_per_block;

    matmul_kernel<<<blocksPerGrid,threads_per_block>>>(A, B, C, n);
}