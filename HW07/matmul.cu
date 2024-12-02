#include <cuda.h>
#include <iostream>
#include <random>
#include "matmul.cuh"

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n)
{
    extern __shared__ unsigned char shared_mem[];
    T *tile_A = reinterpret_cast<T*>(shared_mem);                          // Shared memory for A
    T *tile_B = reinterpret_cast<T*> (&shared_mem[blockDim.x * blockDim.y * sizeof(T)]); // Shared memory for B

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int tile = 0; tile < (n +  blockDim.x - 1)/ blockDim.x; tile++)
    {
        // Load tiles into shared memory
        if (row < n && tile * blockDim.x + threadIdx.x < n)
            tile_A[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + tile * blockDim.x + threadIdx.x];
        else
            tile_A[threadIdx.y * blockDim.x + threadIdx.x] = 0;

        if (col < n && tile * blockDim.y + threadIdx.y < n)
            tile_B[threadIdx.y * blockDim.x + threadIdx.x] = B[(tile * blockDim.y + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y * blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        // Perform multiplication
        for (int k = 0; k < blockDim.x; k++)
        {
            sum += tile_A[threadIdx.y * blockDim.x + k] * tile_B[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// Use template to formulate your answer
template <typename T>
__host__ void matmul(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(T));
    cudaMalloc(&d_B, n * n * sizeof(T));
    cudaMalloc(&d_C, n * n * sizeof(T));

    cudaMemcpy(d_A, A, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(T), cudaMemcpyHostToDevice);

    dim3 block(block_dim, block_dim);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<grid, block, shared_mem_size>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
