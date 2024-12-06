#include <cuda.h>
#include <stdio.h>

__host__ double* calculateDistanceMatrix(const int* hCityX, const int* hCityY, int cityCount, int threadsPerBlock);