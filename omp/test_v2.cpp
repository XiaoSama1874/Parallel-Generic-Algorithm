#include <omp.h>
#include <algorithm>
#include <iostream>

int reduction(int *array, int num_threads, int array_len)
{
    omp_set_num_threads(num_threads);
    int group_len = (array_len + num_threads - 1) / num_threads;
    // First parallel step: local reductions within each group
    #pragma omp parallel for
    for (int i = 0; i < array_len; i += group_len)
    {
        int end = std::min(i + group_len, array_len);
        for (int j = i + 1; j < end; j++)
        {
            array[i] += array[j];
        }
    }

    // Second parallel step: tree-based reduction across groups
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int s = num_threads / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                array[tid * group_len] += array[(tid + s) * group_len];
            }
            #pragma omp barrier
        }
    }
    return array[0];
}

using namespace std;
int main() {
    int array[100];
    fill(array, array + 100, 1);
    cout << reduction(array, 8, 100) << "\n";
    return 0;
}
