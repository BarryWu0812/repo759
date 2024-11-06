#include<cuda.h>
#include<iostream>

__global__ void factorial()
{
    //this adds a value to a variable stored in global memory
    int ans = 1;
    for (int i = 1; i <= threadIdx.x + 1; i++)
        ans *= i;
    std::printf("%d!=%d\n", threadIdx.x + 1, ans);
}
int main()
{
    factorial<<<1,8>>>();
    cudaDeviceSynchronize();
    return 0;
}
