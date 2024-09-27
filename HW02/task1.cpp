#include <iostream>
#include "scan.h"
#include <chrono>
#include <vector>
#include <random>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    //std::cout << "There are " << argc << " arguments:\n";
    int N = std::atoi(argv[1]);
    // std::srand(static_cast<unsigned>(std::time(0)));
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1., 1.);
    
    float* nums = new float[N];
    float* output = new float[N];

    for (int i = 0; i < N; ++i)
    {
        // nums[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0)) - 1.0;
        // nums[i] = i;
        nums[i] = dist(generator);
    }

    start = high_resolution_clock::now();

    scan(nums, output, N);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;

    cout << output[0] << endl;
    cout << output[N-1] << endl;

    delete[] nums;
    delete[] output;
    nums = nullptr;
    output = nullptr;

    return 0;
}