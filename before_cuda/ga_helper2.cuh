#include <cuda.h>
#include <stdio.h>

__host__ double* calculateSumDistances(const int *hPopulation, double *hDistanceMatrix, int populationSize, int cityCount, int threadsPerBlock);