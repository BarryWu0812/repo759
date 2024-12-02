#include <cuda.h>
#include <iostream>
#include <random>
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_int_distribution<int> distint(-1000, 1000);
    std::uniform_real_distribution<float> distf(-1.0, 1.0);
    std::uniform_real_distribution<double> distd(-1.0, 1.0);

    int* A1 = new int[n * n];
    int* B1 = new int[n * n];
    int* C1 = new int[n * n];

    for (unsigned int i = 0; i < n * n; i++)
    {
        A1[i] = distint(generator);
        B1[i] = distint(generator);
        // A1[i] = i;
        // B1[i] = i;
    }

    matmul_1(A1, B1, C1, n, block_dim);

 
    // Floating-point Matrices
    float *A2 = new float[n * n];
    float *B2 = new float[n * n];
    float *C2 = new float[n * n];
    for (unsigned int i = 0; i < n * n; i++)
    {
        A2[i] = distf(generator);
        B2[i] = distf(generator);
        // A2[i] = i + 1;
        // B2[i] = i;
    }

    matmul_2(A2, B2, C2, n, block_dim);

    // Double-precision Matrices
    double *A3 = new double[n * n];
    double *B3 = new double[n * n];
    double *C3 = new double[n * n];
    for (unsigned int i = 0; i < n * n; i++)
    {
        A3[i] = distd(generator);
        B3[i] = distd(generator);
        // A3[i] = i + 2;
        // B3[i] = i;
    }

    matmul_3(A3, B3, C3, n, block_dim);

    delete[] A1;
    delete[] B1;
    delete[] C1;
    delete[] A2;
    delete[] B2;
    delete[] C2;
    delete[] A3;
    delete[] B3;
    delete[] C3;

    return 0;
}
