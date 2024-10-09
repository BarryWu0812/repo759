#include "msort.h"
#include <iostream>
#include <vector>
#include <functional>

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    std::function<void(int*, int, int)> merge = [&](int* arr, int start, int end)
    {
        if (start >= end)
            return;
        int mid = (end + start)/2;
#pragma omp task shared(arr) if (mid - start > threshold)
{
        merge(arr, start, mid);
}
#pragma omp task shared(arr) if (end - (mid + 1) > threshold)
{
        merge(arr, mid + 1, end);
}
#pragma omp taskwait
#pragma omp single
{
        int l = start;
        int r = mid + 1;
        int cnt = 0;
        std::vector<int> tmparr(end - start + 1);
        while (l <= mid && r <= end)
        {
            if (arr[l] < arr[r])
                tmparr[cnt++] = arr[l++];
            else
                tmparr[cnt++] = arr[r++];
        }
        while (l <= mid)
            tmparr[cnt++] = arr[l++];
        while (r <= end)
            tmparr[cnt++] = arr[r++];

        for (int i = 0; i < cnt; i++)
        {
            arr[start + i] = tmparr[i];
        }
}
    };
    merge(arr, 0, n - 1);
}

// int main()
// {
//     int arr[] = {5,3,8,2,7,4,1,9,0};
//     msort(arr, 9, 1);
//     for (int i = 0; i < 9; i++)
//     {
//         std::cout << arr[i] << std::endl;
//     }
//     return 0;
// }