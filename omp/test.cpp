#include <omp.h>
#include <algorithm>
#include <iostream>

int reduction(int *array, int num_threads, int array_len)
{
    omp_set_num_threads(num_threads);
    int group_len = (array_len + num_threads - 1) / num_threads;
    #pragma omp parallel for
    for (int i = 0; i < array_len; i += group_len)
    {
        for (int j = 1; i + j < array_len && j < group_len; j++)
        {
            array[i] += array[i + j];
        }
    }
    #pragma omp parallel for
    for (int tid = 0; tid < num_threads; tid++)
    {
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
int main(){
    int array[100];
    fill(array,array+100,1);
    cout<<reduction(array,8,100);
}