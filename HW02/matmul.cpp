#include "matmul.h"
#include <vector>

void mmul1(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// int main()
// {
//     int n = 3;
//     // double* A = new double[n * n]();
//     double A[9] = {1, 3, 4, 8, 6, 5, 2, 4, 3};
//     double B[9] = {8, 1, 2, 0, 4, 2, 8, 9, 9};
//     double C[9] = {0};
//     double D[9] = {0};
//     double E[9] = {0};
//     // double* C = new double[n * n]();
//     mmul1(A, B, C, n);

//     mmul2(A, B, D, n);

//     mmul3(A, B, E, n);

//     return 0;
// }