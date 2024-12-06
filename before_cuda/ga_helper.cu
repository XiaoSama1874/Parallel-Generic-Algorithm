#include "ga_helper.cuh"
#include <cuda.h>
#include <stdio.h>


__global__ void calculateDistanceMatrixKernel(const int *cityX, const int *cityY, double *dDistanceMatrix, int cityCount) {
    // todo: share memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cityCount * cityCount) {
        int i = idx / cityCount;
        int j = idx % cityCount;
        int dx = cityX[i] - cityX[j];
        int dy = cityY[i] - cityY[j];
        dDistanceMatrix[idx] = sqrt((double)(dx * dx + dy * dy));
    }
}

void calculateDistanceMatrix2(const int *dCityX, const int *dCityY, double *dDistanceMatrix, int cityCount, int threadsPerBlock) {
    int num_blocks = (cityCount * cityCount + threadsPerBlock - 1) / threadsPerBlock;
    calculateDistanceMatrixKernel<<<num_blocks, threadsPerBlock>>>(dCityX, dCityY, dDistanceMatrix, cityCount);
    cudaDeviceSynchronize();
}

__host__  double* calculateDistanceMatrix(const int *hCityX, const int *hCityY, int cityCount, int threadsPerBlock) {
    int *dCityX, *dCityY;
    double *hDistanceMatrix, *dDistanceMatrix;
    hDistanceMatrix = new double[cityCount * cityCount];
    cudaMalloc((void **)&dCityX, cityCount * sizeof(int));
    cudaMalloc((void **)&dCityY, cityCount * sizeof(int));
    cudaMalloc((void **)&dDistanceMatrix, cityCount * cityCount * sizeof(double));
    cudaMemcpy(dCityX, hCityX, cityCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCityY, hCityY, cityCount * sizeof(int), cudaMemcpyHostToDevice);
    calculateDistanceMatrix2(dCityX, dCityY, dDistanceMatrix, cityCount, threadsPerBlock);
    cudaMemcpy(hDistanceMatrix, dDistanceMatrix, cityCount * cityCount * sizeof(double), cudaMemcpyDeviceToHost);
    return hDistanceMatrix;
}