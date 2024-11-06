#include <cuda.h>
#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n)
{
    int whichEntry = threadIdx.x + blockIdx.x * blockDim.x;
    if (whichEntry < n)
        b[whichEntry] = a[whichEntry] * b[whichEntry];
}