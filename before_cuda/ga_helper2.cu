#include "ga_helper.cuh"
#include <cuda.h>
#include <stdio.h>


__global__ void calculateSumDistancesKernel(int *dPopulation, double *dDistanceMatrix, double *dSumDistances, int populationSize, int cityCount) {
    // todo: share memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        int* path = dPopulation + idx * cityCount;        
        double totalDistance = dDistanceMatrix[path[cityCount - 1]*cityCount + path[0]];
        for (int i = 0; i < cityCount - 1; ++i) {
            totalDistance += dDistanceMatrix[path[i]*cityCount + path[i + 1]];
        }
        dSumDistances[idx] = totalDistance;
    }
}

void calculateSumDistances2(int *dPopulation, double *dDistanceMatrix, double *dSumDistances, int populationSize, int cityCount, int threadsPerBlock) {
    int num_blocks = (populationSize + threadsPerBlock - 1) / threadsPerBlock;
    calculateSumDistancesKernel<<<num_blocks, threadsPerBlock>>>(dPopulation, dDistanceMatrix, dSumDistances, populationSize, cityCount);
    cudaDeviceSynchronize();
}

__host__  double* calculateSumDistances(const int *hPopulation, double *hDistanceMatrix, int populationSize, int cityCount, int threadsPerBlock) {
    double *hSumDistances;
    hSumDistances = new double[populationSize];
    double *dSumDistances;
    int *dPopulation;
    double *dDistanceMatrix;
    hDistanceMatrix = new double[cityCount * cityCount];
    cudaMalloc((void **)&dSumDistances, populationSize * sizeof(double));
    cudaMalloc((void **)&dPopulation, populationSize * cityCount * sizeof(int));
    cudaMemcpy(dPopulation, hPopulation, populationSize * cityCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dDistanceMatrix, cityCount * cityCount * sizeof(double));
    cudaMemcpy(dDistanceMatrix, hDistanceMatrix, cityCount * cityCount * sizeof(double), cudaMemcpyHostToDevice);
    calculateSumDistances2(dPopulation, dDistanceMatrix, dSumDistances, populationSize, cityCount, threadsPerBlock);
    cudaMemcpy(hSumDistances, dSumDistances, populationSize  * sizeof(double), cudaMemcpyDeviceToHost);
    return hSumDistances;
}