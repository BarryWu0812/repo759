#include <iostream>
#include "convolution.h"
#include <chrono>
#include <random>
#include <omp.h>

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
    // std::srand(static_cast<unsigned>(std::time(0)));
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> distn(-10., 10.);
    std::uniform_real_distribution<float> distm(-1., 1.);
    omp_set_num_threads(std::atoi(argv[2]));
    float* image = new float[n * n];

    for (int i = 0; i < n * n; i++)
    {
        // image[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 20.0)) - 10.0;
        // image[i] = i;
        image[i] = distn(generator);

    }

    std::size_t m = 3;
    float* mask = new float[m * m];
    for (int i = 0; i < m * m; i++)
    {
        // mask[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0)) - 1.0;
        // mask[i] = i;
        mask[i] = distm(generator);
    }

    start = high_resolution_clock::now();

    // Perform convolution
    float* output = new float[n * n];
    convolve(image, output, n, mask, m);
    // printf("Actual number of threads used in this other parallel region: %d\n", omp_get_num_threads());

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);

    cout << output[0] << endl;
    cout << output[n * n - 1] << endl;
    cout << duration_sec.count() << endl;


    

    delete[] image;
    delete[] mask;
    delete[] output;
    image = nullptr;
    mask = nullptr;
    output = nullptr;

    return 0;
}