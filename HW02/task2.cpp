#include <iostream>
#include "convolution.h"
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    std::size_t n = std::atoi(argv[1]);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    std::srand(static_cast<unsigned>(std::time(0)));
    float* image = new float[n * n];

    for (int i = 0; i < n * n; i++)
    {
        image[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 20.0)) - 10.0;
        // image[i] = i;

    }

    std::size_t m = std::atoi(argv[2]);
    float* mask = new float[m * m];
    for (int i = 0; i < m * m; i++)
    {
        mask[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0)) - 1.0;
        // mask[i] = i;

    }

    start = high_resolution_clock::now();

    // Perform convolution
    float* output = new float[n * n];
    convolve(image, output, n, mask, m);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout << duration_sec.count() << endl;

    cout << output[0] << endl;
    cout << output[n * n - 1] << endl;

    

    delete[] image;
    delete[] mask;
    delete[] output;
    image = nullptr;
    mask = nullptr;
    output = nullptr;

    return 0;
}