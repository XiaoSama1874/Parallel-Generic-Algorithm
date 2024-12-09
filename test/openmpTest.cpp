#include <iostream>
#include <omp.h>

int main() {
    omp_set_dynamic(0); // 禁用动态线程调整
    omp_set_num_threads(4); // 显式设置线程数

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        #pragma omp critical
        std::cout << "Thread ID: " << thread_id << ", Total Threads: " << num_threads << std::endl;
    }

    return 0;
}
