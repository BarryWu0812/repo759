#include<cuda.h>
#include<iostream>
#include <random>
__global__ void simpleKernel(int* data, int a)
{
    //this adds a value to a variable stored in global memory
    int whichEntry = threadIdx.x + blockIdx.x * blockDim.x;
    // std::printf("whichEntry = %d, threadIdx = %d, blockIdx = %d, blockDim = %d\n", whichEntry, threadIdx.x, blockIdx.x, blockDim.x);
    if (whichEntry < 16)
        data[whichEntry] = a * threadIdx.x + blockIdx.x;
}
int main()
{
    const int numElems = 512;
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_int_distribution<int> dist(-10, 10);
    int a = dist(generator);
    int hA[numElems], *dA;
    //allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void**)&dA, sizeof(int) * numElems);
    cudaMemset(dA, 0, numElems * sizeof(int));
    //invoke GPU kernel, with one block that has four threads
    simpleKernel<<<2,8>>>(dA, a);
    //bring the result back from the GPU into the hostArray
    cudaMemcpy(hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
    //print out the result to confirm that things are looking good
    // std::cout << "Values stored in hA: " << std::endl;
    for (int i = 0; i < numElems; i++)
        std::cout << hA[i] << "\t";
    std::cout << std::endl;
    //release the memory allocated on the GPU
    cudaFree(dA);
    return 0;
}
