#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    int n = 1000;
    cout << n << endl;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C1 = new double[n * n]();
    double* C2 = new double[n * n]();
    double* C3 = new double[n * n]();
    double* C4 = new double[n * n]();
    std::vector<double> A2(n * n);
    std::vector<double> B2(n * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = i + j;
            B[i * n + j] = i + j;
            A2[i * n + j] = i + j;
            B2[i * n + j] = i + j;
        }
    }

    start = high_resolution_clock::now();
    mmul1(A, B, C1, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;
    cout << C1[n * n - 1] << endl;
    start = high_resolution_clock::now();
    mmul2(A, B, C2, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;
    cout << C2[n * n - 1] << endl;
    start = high_resolution_clock::now();
    mmul3(A, B, C3, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;
    cout << C3[n * n - 1] << endl;
    start = high_resolution_clock::now();
    mmul4(A2, B2, C4, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;
    cout << C4[n * n - 1] << endl;

    return 0;
}