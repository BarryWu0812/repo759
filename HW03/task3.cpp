#include <iostream>
#include "msort.h"
#include <random>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    omp_set_num_threads(t);
    int ts = std::atoi(argv[3]);
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_int_distribution<int> dist(-1000, 1000);
    int* arr = new int[n];
    for (int i = 0; i < n; i++)
    {
        // image[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 20.0)) - 10.0;
        // image[i] = i;
        arr[i] = dist(generator);

    }
    start = high_resolution_clock::now();
    msort(arr, n, ts);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    std::cout << arr[0] << std::endl;
    std::cout << arr[n-1] << std::endl;
    std::cout << duration_sec.count() << std::endl;

    // for (int i = 0; i < n; i++)
    // {
    //     std::cout << arr[i] << std::endl;
    // }
    return 0;
}