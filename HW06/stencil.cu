#include <cuda.h>
#include <iostream>
#include "stencil.cuh"


// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
    extern __shared__ float shared_mem[];
    float* shared_image = shared_mem;
    float* shared_mask = shared_mem + blockDim.x + 2 * R;
    float* shared_output = shared_mask + (2 * R + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < (2 * R + 1))
    {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
        // printf("Mask[%d] = %f\n", threadIdx.x, shared_mask[threadIdx.x]);
    }

    if (idx < n)
    {
        shared_image[threadIdx.x + R] = image[idx];
        // printf("Image[%d] = %f\n", idx, shared_image[threadIdx.x + R]);
    }
    else
    {
        shared_image[threadIdx.x + R] = 1;
        // printf("Image[%d] = 1 (boundary)\n", idx);
    }

    if (threadIdx.x < R) 
    {
        if (idx >= R)
            shared_image[threadIdx.x] = image[idx - R];
        else
            shared_image[threadIdx.x] = 1;

        // printf("Left Halo[%d] = %f\n", threadIdx.x, shared_image[threadIdx.x]);
    }

    if (threadIdx.x >= blockDim.x - R)
    {
        if (idx + R < n)
            shared_image[threadIdx.x + R + R] = image[idx + R];
        else
            shared_image[threadIdx.x + R + R] = 1;
        // printf("Right Halo[%d] = %f\n", threadIdx.x + R + blockDim.x, shared_image[threadIdx.x + R + blockDim.x]);
    }
    __syncthreads();

    if (idx < n)
    {
        // printf("bIdx = %d, tIdx = %d, =====Mask[%d] = %f\n", blockIdx.x, threadIdx.x, threadIdx.x, shared_mask[threadIdx.x]);
        float sum = 0;
        for (int j = -static_cast<int>(R); j <= static_cast<int>(R); j++)
        {
            // printf("Thread %d: shared_image[%d] = %f, shared_mask[%d] = %f\n", 
            //    threadIdx.x, threadIdx.x + R + j, shared_image[threadIdx.x + R + j], j + R, shared_mask[j + R]);
            sum += shared_image[threadIdx.x + R + j] * shared_mask[j + R];
        }
        shared_output[threadIdx.x] = sum;
        // printf("Thread %d: Computed sum = %f\n", threadIdx.x, sum);
    }
    __syncthreads();
    if (idx < n)
        output[idx] = shared_output[threadIdx.x];

}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    unsigned int blocksPerGrid = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_memory_size = (threads_per_block + 2 * R) * sizeof(float) + (2 * R + 1) * sizeof(float) + threads_per_block * sizeof(float);
    stencil_kernel<<<blocksPerGrid, threads_per_block, shared_memory_size>>>(image, mask, output, n, R);

    cudaDeviceSynchronize();
}
