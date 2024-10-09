#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    int n = std::atoi(argv[1]);
    // cout << n << endl;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    omp_set_num_threads(std::atoi(argv[2]));

    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n]();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = i + j;
            B[i * n + j] = i + j;
        }
    }

    start = high_resolution_clock::now();
    mmul(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << C[0] << endl;
    cout << C[n * n - 1] << endl;
    cout << duration_sec.count() << endl;

    return 0;
}