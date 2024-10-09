#include "convolution.h"
#include <iostream>
#include <vector>


float get_f(const float *image, int x, int y, int i, int j, std::size_t n, std::size_t m)
{
    if ((0 > x + i - (m - 1)/2  || x + i - (m - 1)/2 >= n) && (0 > y + j - (m - 1)/2  || y + j - (m - 1)/2 >= n))
    {
        // cout << "0" << endl;
        return 0;
    }
    else if (0 > x + i - (m - 1)/2  || x + i - (m - 1)/2 >= n)
    {
        // cout << "1" << endl;
        return 1;
    }
    else if (0 > y + j - (m - 1)/2  || y + j - (m - 1)/2 >= n)
    {
        // cout << "2" << endl;
        return 1;
    }
    else
    {
        // cout << "3" << endl;
        return image[n * (x + i - (m - 1)/2) + (y + j - (m - 1)/2)];
    }
}

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
#pragma omp parallel for collapse(2)
    for (int x = 0; x <= n - 1; x++)
    {
        for (int y = 0; y <= n - 1; y++)
        {
            float sum = 0;
            for (int i = 0; i <= m - 1; i++)
            {
                for (int j = 0; j <= m - 1; j++)
                {
                    float pad_f = get_f(image, x, y, i, j, n, m);
                    // cout << "pad_f: "<< pad_f << endl;
                    
                    sum += mask[i * m + j] * pad_f;
                }
            }
            output[x * n + y] = sum;
            //cout << x * n + y << ": " << output[x * n + y] << endl;
        }
    }    
}
// int main() {
//     // Example image f (4x4 matrix)
//     float f[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};

//     // Example convolution kernel omega (3x3 matrix)
//     float omega[9] = { 0, 0, 1, 0, 1, 0, 1, 0, 0};

//     std::size_t n = 4;
//     std::size_t m = 3;
//     // Perform convolution
//     float* output = new float[n * n];
//     convolve(f, output, n, omega, m);

//     return 0;
// }