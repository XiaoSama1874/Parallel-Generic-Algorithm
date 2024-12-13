#include <omp.h>

int reduction(int* array, int num_threads, int array_len) {
    omp_set_num_threads(num_threads);
    int group_len = (array_len + num_threads-1)/num_threads;
    #pragma omp parallel for
    for (int i = 0; i < array_len; i += group_len) {
    for (j = 1; i + j < array_len && j < group_len; j++) {
    array[i] += array[i + j];
    }
    }
    #pragma omp parallel for
    for (int tid = 0; tid < num_threads; tid++) {
    for (int s=num_threads/2; s>0; s>>=1) {
    if (tid < s) {
    sdata[tid*group_len] += sdata[(tid + s)*group_len];
    }
    #pragma omp barrier
    }
    }
    return array[0];
}