#include <iostream>
#include "scan.h"
#include <chrono>
#include <vector>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    //std::cout << "There are " << argc << " arguments:\n";
    int N = std::atoi(argv[1]);
    std::srand(static_cast<unsigned>(std::time(0)));

    start = high_resolution_clock::now();
    
    std::vector<float> nums;
    for (int i = 0; i < N; ++i) {
        float random_float = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0)) - 1.0;
        nums.push_back(random_float);
    }

    // Print the generated random numbers
    for (float num : nums) {
        std::cout << num << " ";
    }

    cout << '\n';
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "Total time: " << duration_sec.count() << "ms\n";
    return 0;
}