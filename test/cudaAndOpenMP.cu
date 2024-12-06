#include <iostream>
#include <omp.h>
#include <cuda.h>

__global__ void gpu_square(float* d_array, int start, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        d_array[idx] *= d_array[idx];
    }
}

void cpu_square(float* h_array, int start, int end) {
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        std::printf("thread: %d\n",omp_get_thread_num());
        h_array[i] *= h_array[i];
    }
}

int main() {
    // const int N = 1 << 20;  // 数组大小
    const int N = 10;  // 数组大小

    const int GPU_PART = N / 2;  // 一半数据在 GPU 上计算
    const int CPU_PART = N - GPU_PART;  // 另一半数据在 CPU 上计算

    omp_set_dynamic(0); // 禁用动态线程调整
   omp_set_num_threads(4);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // 获取线程号
        int num_threads = omp_get_num_threads(); // 获取总线程数
        #pragma omp critical
        std::cout << "Thread ID: " << thread_id << ", Total Threads: " << num_threads << std::endl;
    }



    float* h_array = new float[N];
    for (int i = 0; i < N; i++) {
        h_array[i] = static_cast<float>(i);
    }

    // CUDA 部分
    float* d_array;
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (GPU_PART + threadsPerBlock - 1) / threadsPerBlock;

    gpu_square<<<blocksPerGrid, threadsPerBlock>>>(d_array, 0, GPU_PART);
    cudaDeviceSynchronize();

    // OpenMP 部分
    cpu_square(h_array, GPU_PART, N);

    // 合并结果
    cudaMemcpy(h_array, d_array, GPU_PART * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_array);

    cudaDeviceSynchronize();

    // 输出部分结果
    for (int i = 0; i < 5; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << "\n";
    for (int i = GPU_PART; i < GPU_PART + 5; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << "\n";

    delete[] h_array;
    return 0;
}
