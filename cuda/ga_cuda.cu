#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess)                                                                   \
        {                                                                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

using namespace std;

int CITY_COUNT = 100;                 // city scale
int POPULATION_SIZE = 1000;           // population scale
int GENERATIONS = 1000;               // iteration times
const double MUTATION_RATE = 0.1;     // mutation rate
const double ELITISM_THRESHOLD = 0.2; // elite
bool PRINT_EACH_ITERATION = true;     // debug
const int TOURNAMENT_SIZE = 5;        // For tournament selection
struct City
{
    int x, y;
};

// Kernel to initialize curand states
__global__ void initRNGStates(curandState_t *states, unsigned long seed, int POPULATION_SIZE)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// GPU kernel: compute fitness
__global__ void computeFitnessKernel(const int *population, const double *distanceMatrix, double *fitness, int POPULATION_SIZE, int CITY_COUNT)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        double totalDist = 0.0;
        const int base = idx * CITY_COUNT;
        for (int i = 0; i < CITY_COUNT - 1; i++)
        {
            int c1 = population[base + i];
            int c2 = population[base + i + 1];
            totalDist += distanceMatrix[c1 * CITY_COUNT + c2];
        }
        // return to start
        int c1 = population[base + CITY_COUNT - 1];
        int c2 = population[base];
        totalDist += distanceMatrix[c1 * CITY_COUNT + c2];

        fitness[idx] = 1.0 / totalDist;
    }
}

// Kernel: Tournament selection
__global__ void tournamentSelectionKernel(const double *fitness, int *selectedIndices, int POPULATION_SIZE, int tournamentSize, curandState_t *states)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        curandState_t localState = states[idx];
        double bestFit = -1.0;
        int bestIdx = -1;
        for (int i = 0; i < tournamentSize; i++)
        {
            int cand = curand(&localState) % POPULATION_SIZE;
            double f = fitness[cand];
            if (f > bestFit)
            {
                bestFit = f;
                bestIdx = cand;
            }
        }
        selectedIndices[idx] = bestIdx;
        states[idx] = localState;
    }
}

// Kernel: OX Crossover
__global__ void crossoverKernel(const int *population, const int *selectedIndices,
                                int *newPopulation, curandState_t *states, int halfNonElite, int CITY_COUNT)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Handle pairs: idx-th pair corresponds to indices 2*idx and 2*idx+1
    if (idx < halfNonElite)
    {
        int parent1Idx = selectedIndices[2 * idx];
        int parent2Idx = selectedIndices[2 * idx + 1];
        const int *p1 = &population[parent1Idx * CITY_COUNT];
        const int *p2 = &population[parent2Idx * CITY_COUNT];

        int *child1 = &newPopulation[(2 * idx) * CITY_COUNT];
        int *child2 = &newPopulation[(2 * idx + 1) * CITY_COUNT];

        curandState_t localState = states[idx];
        int start = curand(&localState) % CITY_COUNT;
        int end = curand(&localState) % CITY_COUNT;
        if (start > end)
        {
            int tmp = start;
            start = end;
            end = tmp;
        }

        // OX crossover:
        // child1
        for (int i = start; i <= end; i++)
            child1[i] = p1[i];
        {
            int cpos = (end + 1) % CITY_COUNT;
            for (int i = 0; i < CITY_COUNT; i++)
            {
                int gene = p2[i];
                bool found = false;
                for (int j = start; j <= end; j++)
                {
                    if (child1[j] == gene)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    child1[cpos] = gene;
                    cpos = (cpos + 1) % CITY_COUNT;
                }
            }
        }

        // child2
        for (int i = start; i <= end; i++)
            child2[i] = p2[i];
        {
            int cpos = (end + 1) % CITY_COUNT;
            for (int i = 0; i < CITY_COUNT; i++)
            {
                int gene = p1[i];
                bool found = false;
                for (int j = start; j <= end; j++)
                {
                    if (child2[j] == gene)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    child2[cpos] = gene;
                    cpos = (cpos + 1) % CITY_COUNT;
                }
            }
        }

        states[idx] = localState;
    }
}

// Kernel: Mutation
__global__ void mutationKernel(int *population, double mutationRate, curandState_t *states, int POPULATION_SIZE, int CITY_COUNT)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        curandState_t localState = states[idx];
        float r = curand_uniform(&localState);
        if (r < mutationRate)
        {
            int c1 = curand(&localState) % CITY_COUNT;
            int c2 = curand(&localState) % CITY_COUNT;
            int base = idx * CITY_COUNT;
            int temp = population[base + c1];
            population[base + c1] = population[base + c2];
            population[base + c2] = temp;
        }
        states[idx] = localState;
    }
}

City *initializeCities()
{
    City *cities = new City[CITY_COUNT];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 100);
    for (int i = 0; i < CITY_COUNT; ++i)
    {
        cities[i] = {dist(gen), dist(gen)};
    }
    return cities;
}

double *computeDistanceMatrix(const City *cities)
{
    double *distMat = new double[CITY_COUNT * CITY_COUNT];
    for (int i = 0; i < CITY_COUNT; ++i)
    {
        for (int j = 0; j < CITY_COUNT; ++j)
        {
            if (i == j)
                distMat[i * CITY_COUNT + j] = 0.0;
            else
            {
                double dx = cities[i].x - cities[j].x;
                double dy = cities[i].y - cities[j].y;
                distMat[i * CITY_COUNT + j] = sqrt(dx * dx + dy * dy);
            }
        }
    }
    return distMat;
}

__global__ void initPopulationKernel(int *d_population, const int *d_template, curandState_t *d_states, int POPULATION_SIZE, int CITY_COUNT)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        curandState_t localState = d_states[idx];

        int startPos = idx * CITY_COUNT;
        // Copy template
        for (int i = 0; i < CITY_COUNT; i++)
        {
            d_population[startPos + i] = d_template[i];
        }

        // Shuffle (Fisher-Yates)
        for (int i = CITY_COUNT - 1; i > 0; i--)
        {
            int r = curand(&localState) % (i + 1);
            int temp = d_population[startPos + i];
            d_population[startPos + i] = d_population[startPos + r];
            d_population[startPos + r] = temp;
        }

        d_states[idx] = localState;
    }
}

void initializePopulationOnGPU(int *d_population, curandState_t *d_states)
{
    // Create template array on CPU
    std::vector<int> templatePath(CITY_COUNT);
    std::iota(templatePath.begin(), templatePath.end(), 0);

    // Copy template to GPU
    int *d_template;
    CUDA_CHECK(cudaMalloc((void **)&d_template, CITY_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_template, templatePath.data(), CITY_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (POPULATION_SIZE + threads - 1) / threads;
    initPopulationKernel<<<blocks, threads>>>(d_population, d_template, d_states, POPULATION_SIZE, CITY_COUNT);
    cudaDeviceSynchronize();

    cudaFree(d_template);
}

double calculatePathDistance(const int *path, const double *distMat)
{
    double totalDist = 0.0;
    for (int i = 0; i < CITY_COUNT - 1; i++)
    {
        totalDist += distMat[path[i] * CITY_COUNT + path[i + 1]];
    }
    totalDist += distMat[path[CITY_COUNT - 1] * CITY_COUNT + path[0]];
    return totalDist;
}

__host__ void geneticAlgorithm(City* cities, double* h_distMat)
{
    // Allocate distance matrix on GPU
    double *d_distanceMatix;
    CUDA_CHECK(cudaMalloc((void **)&d_distanceMatix, CITY_COUNT * CITY_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_distanceMatix, h_distMat, CITY_COUNT * CITY_COUNT * sizeof(double), cudaMemcpyHostToDevice));

    int *d_population;
    int *d_newPopulation;
    double *d_fitness;
    int *d_selectedIndices;

    CUDA_CHECK(cudaMalloc((void **)&d_population, POPULATION_SIZE * CITY_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_newPopulation, POPULATION_SIZE * CITY_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_selectedIndices, POPULATION_SIZE * sizeof(int)));

    // Setup RNG
    curandState_t *d_states;
    CUDA_CHECK(cudaMalloc((void **)&d_states, POPULATION_SIZE * sizeof(curandState_t)));
    int threads = 256;
    int blocks = (POPULATION_SIZE + threads - 1) / threads;
    initRNGStates<<<blocks, threads>>>(d_states, 1234, POPULATION_SIZE);

    // Initialize population on GPU
    initializePopulationOnGPU(d_population, d_states);

    int eliteCount = (int)(ELITISM_THRESHOLD * POPULATION_SIZE);
    int nonEliteCount = POPULATION_SIZE - eliteCount;
    int halfNonElite = nonEliteCount / 2;
    int cb = (nonEliteCount + threads - 1) / threads;
    int mutation_blocks = (nonEliteCount + threads - 1) / threads;


    double *result_check_fitness = new double[POPULATION_SIZE];

    cout << "finish initialization" << endl;
    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        // Compute fitness
        computeFitnessKernel<<<blocks, threads>>>(d_population, d_distanceMatix, d_fitness, POPULATION_SIZE, CITY_COUNT);
        cudaDeviceSynchronize();

        // Copy fitness to host and do elitism
        double *h_fitness = new double[POPULATION_SIZE];
        CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        vector<pair<double, int>> fitIdx(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++)
            fitIdx[i] = {h_fitness[i], i};
        sort(fitIdx.begin(), fitIdx.end(), greater<>());

        int *h_eliteIndices = new int[eliteCount];
        for (int i = 0; i < eliteCount; i++)
        {
            h_eliteIndices[i] = fitIdx[i].second;
        }

        int *h_popBuffer = new int[POPULATION_SIZE * CITY_COUNT];
        CUDA_CHECK(cudaMemcpy(h_popBuffer, d_population, POPULATION_SIZE * CITY_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

        int *h_eliteBuffer = new int[eliteCount * CITY_COUNT];
        for (int e = 0; e < eliteCount; e++)
        {
            int idx = h_eliteIndices[e];
            memcpy(&h_eliteBuffer[e * CITY_COUNT], &h_popBuffer[idx * CITY_COUNT], CITY_COUNT * sizeof(int));
        }

        CUDA_CHECK(cudaMemcpy(d_newPopulation, h_eliteBuffer, eliteCount * CITY_COUNT * sizeof(int), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

        delete[] h_eliteIndices;
        delete[] h_fitness;
        delete[] h_popBuffer;
        delete[] h_eliteBuffer;

        // Selection
        tournamentSelectionKernel<<<blocks, threads>>>(d_fitness, d_selectedIndices, POPULATION_SIZE, TOURNAMENT_SIZE, d_states);
        cudaDeviceSynchronize();

        // Crossover
        crossoverKernel<<<cb, threads>>>(d_population, d_selectedIndices, d_newPopulation + eliteCount * CITY_COUNT, d_states, halfNonElite, CITY_COUNT);
        cudaDeviceSynchronize();

        // Mutation
        mutationKernel<<<mutation_blocks, threads>>>(d_newPopulation + (eliteCount * CITY_COUNT), MUTATION_RATE, d_states, nonEliteCount, CITY_COUNT);
        cudaDeviceSynchronize();

        int *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;

        if (PRINT_EACH_ITERATION && (gen % 50 == 0))
        {
            CUDA_CHECK(cudaMemcpy(result_check_fitness, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            double bestFit = result_check_fitness[0];
            for (int i = 1; i < POPULATION_SIZE; i++)
            {
                if (result_check_fitness[i] > bestFit)
                {
                    bestFit = result_check_fitness[i];
                }
            }
            double bestDistance = 1.0 / bestFit;
            cout << "Generation " << gen << ": Best Distance = " << bestDistance << endl;
        }
    }

    // Final calculation
    computeFitnessKernel<<<blocks, threads>>>(d_population, d_distanceMatix, d_fitness, POPULATION_SIZE, CITY_COUNT);
    cudaDeviceSynchronize();

    double *h_fitness = new double[POPULATION_SIZE];
    CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    int bestIdx = 0;
    double bestFit = h_fitness[0];
    for (int i = 1; i < POPULATION_SIZE; i++)
    {
        if (h_fitness[i] > bestFit)
        {
            bestFit = h_fitness[i];
            bestIdx = i;
        }
    }

    int *h_solution = new int[CITY_COUNT];
    CUDA_CHECK(cudaMemcpy(h_solution, d_population + bestIdx * CITY_COUNT, CITY_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

    double bestDistance = 1.0 / bestFit;
    cout << "Best distance: " << bestDistance << endl;
    cout << "Best path:" << endl;
    for (int i = 0; i < CITY_COUNT; i++)
        cout << h_solution[i] << " ";
    cout << "\n";

    // Cleanup
    delete[] h_solution;
    delete[] h_fitness;
    delete[] result_check_fitness; 
    // Removed delete[] cities; from here to avoid double free.

    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_fitness);
    cudaFree(d_selectedIndices);
    cudaFree(d_distanceMatix);
    cudaFree(d_states);
}


int main(int argc, char* argv[])
{
    if (argc > 1) CITY_COUNT = stoi(argv[1]);
    if (argc > 2) POPULATION_SIZE = stoi(argv[2]);
    if (argc > 3) GENERATIONS = stoi(argv[3]);
    if (argc > 4) {
        string val = argv[4];
        if (val == "false" || val == "0") {
            PRINT_EACH_ITERATION = false;
        } else {
            PRINT_EACH_ITERATION = true;
        }
    }

    cout << "CITY_COUNT=" << CITY_COUNT << ", POPULATION_SIZE=" << POPULATION_SIZE
         << ", GENERATIONS=" << GENERATIONS
         << ", PRINT_EACH_ITERATION=" << (PRINT_EACH_ITERATION ? "true" : "false") << endl;

    City *cities = initializeCities();
    double *h_distMat = computeDistanceMatrix(cities);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    float ms = 0.0f;

    geneticAlgorithm(cities, h_distMat);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cout << "Time (ms): " << ms << endl;

    delete[] h_distMat;
    delete[] cities;

    return 0;
}
