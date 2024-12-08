#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess)                                                                   \
        {                                                                                         \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
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
// population: POPULATION_SIZE x CITY_COUNT
// distanceMatrix: CITY_COUNT * CITY_COUNT
// fitness: POPULATION_SIZE
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
// Given current population and fitness, select parents indices and write them into a parentIndex array
__global__ void tournamentSelectionKernel(const double *fitness, int *selectedIndices, int POPULATION_SZIE, int tournamentSize, curandState_t *states)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SZIE)
    {
        curandState_t localState = states[idx];
        double bestFit = -1.0;
        int bestIdx = -1;
        for (int i = 0; i < tournamentSize; i++)
        {
            int cand = curand(&localState) % POPULATION_SZIE;
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


__global__ void crossoverKernel(const int *population, const int *selectedIndices,
                                int *newPopulation, curandState_t *states, int halfNonElite, int CITY_COUNT, double mutationRate)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
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

        // OX crossover for child1
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

        // OX crossover for child2
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

        // apply mutation to both children
        float r = curand_uniform(&localState);
        if (r < mutationRate)
        {
            int c1 = curand(&localState) % CITY_COUNT;
            int c2 = curand(&localState) % CITY_COUNT;
            int temp = child1[c1];
            child1[c1] = child1[c2];
            child1[c2] = temp;
        }

        r = curand_uniform(&localState);
        if (r < mutationRate)
        {
            int c1 = curand(&localState) % CITY_COUNT;
            int c2 = curand(&localState) % CITY_COUNT;
            int temp = child2[c1];
            child2[c1] = child2[c2];
            child2[c2] = temp;
        }

        states[idx] = localState;
    }
}


// CPU: initialize cities
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

// CPU: compute distance matrix
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

__global__ void initPopulationKernel(int *d_population, const int *d_template, curandState_t *d_states,int POPULATION_SIZE ,int CITY_COUNT)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < POPULATION_SIZE)
    {
        curandState_t localState = d_states[idx];

        int startPos = idx * CITY_COUNT;
        // Copy template into individual's route
        for (int i = 0; i < CITY_COUNT; i++)
        {
            d_population[startPos + i] = d_template[i];
        }

        // Fisher–Yates shuffle
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
    cudaMalloc((void **)&d_template, CITY_COUNT * sizeof(int));
    cudaMemcpy(d_template, templatePath.data(), CITY_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to shuffle and create the population
    int threads = 256;
    int blocks = (POPULATION_SIZE + threads - 1) / threads;
    initPopulationKernel<<<blocks, threads>>>(d_population, d_template, d_states, POPULATION_SIZE,CITY_COUNT);
    cudaDeviceSynchronize();

    cudaFree(d_template);
}

// CPU helper to create initial population (random permutations)
void initializePopulationCPU(int *h_population)
{
    std::vector<int> path(CITY_COUNT);
    iota(path.begin(), path.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        std::shuffle(path.begin(), path.end(), gen);
        for (int j = 0; j < CITY_COUNT; j++)
        {
            h_population[i * CITY_COUNT + j] = path[j];
        }
    }
}

// CPU: find best after finishing
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

int main(int argc, char* argv[]) {

    if (argc > 1) CITY_COUNT = stoi(argv[1]);
    if (argc > 2) POPULATION_SIZE = stoi(argv[2]);
    if (argc > 3) GENERATIONS = stoi(argv[3]);
    if (argc > 4) {
        string val = argv[4];
        // Convert string to bool
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


    // GPU memory
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

    // Setup random RNG
    curandState_t *d_states;
    CUDA_CHECK(cudaMalloc((void **)&d_states, POPULATION_SIZE * sizeof(curandState_t)));
    initRNGStates<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_states, 1234, POPULATION_SIZE);

    // Initialize population on gpu
    initializePopulationOnGPU(d_population, d_states);

  

    double *result_check_fitness = new double[POPULATION_SIZE];

    cout << "finish initialization" << endl;
    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        // cout<<"generation: "<<gen<<endl;

        // Compute fitness
        computeFitnessKernel<<<blocks, threads>>>(d_population, d_distanceMatix, d_fitness, POPULATION_SIZE, CITY_COUNT);
        cudaDeviceSynchronize();
        // cout<<"finish computeFitnessKernel"<<endl;

        // Elitism: sort on cpu because of the small population size
        double *h_fitness = new double[POPULATION_SIZE];
        CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        vector<pair<double, int>> fitIdx(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++)
            fitIdx[i] = {h_fitness[i], i};
        sort(fitIdx.begin(), fitIdx.end(), greater<>());
        // cout<<"finish sort"<<h_fitness[0]<<endl;

        // Copy elites to newPopulation on GPU
        // This can be done on CPU by copying indices and using cudaMemcpy
        int *h_eliteIndices = new int[eliteCount];
        for (int i = 0; i < eliteCount; i++)
        {
            h_eliteIndices[i] = fitIdx[i].second;
        }

        // Copy the entire population from device to host
        int *h_popBuffer = new int[POPULATION_SIZE * CITY_COUNT];
        CUDA_CHECK(cudaMemcpy(h_popBuffer, d_population, POPULATION_SIZE * CITY_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

        // Now extract elites from h_popBuffer (on CPU)
        int *h_eliteBuffer = new int[eliteCount * CITY_COUNT];
        for (int e = 0; e < eliteCount; e++)
        {
            int idx = h_eliteIndices[e];
            memcpy(&h_eliteBuffer[e * CITY_COUNT], &h_popBuffer[idx * CITY_COUNT], CITY_COUNT * sizeof(int));
        }

        // Copy elites back to the GPU
        CUDA_CHECK(cudaMemcpy(d_newPopulation, h_eliteBuffer, eliteCount * CITY_COUNT * sizeof(int), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

        delete[] h_eliteIndices;
        delete[] h_fitness;
        delete[] h_popBuffer;
        delete[] h_eliteBuffer;

        // cout<<"start selection"<<endl;
        tournamentSelectionKernel<<<blocks, threads>>>(d_fitness, d_selectedIndices, POPULATION_SIZE, TOURNAMENT_SIZE, d_states);
        cudaDeviceSynchronize();
        // cout<<"finish selection"<<endl;

        // cout<<"start crossover"<<endl;
       crossoverKernel<<<cb, threads>>>(d_population, d_selectedIndices, d_newPopulation + eliteCount * CITY_COUNT, d_states, halfNonElite, CITY_COUNT, MUTATION_RATE);
        // cout<<"finish crossover"<<endl;

        cudaDeviceSynchronize();


        int *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;

        // Debug
        if (PRINT_EACH_ITERATION && (gen % 50 == 0))
        {

            CUDA_CHECK(cudaMemcpy(result_check_fitness, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

            // Find the best individual on the CPU
            double bestFit = result_check_fitness[0];
            for (int i = 1; i < POPULATION_SIZE; i++)
            {
                if (result_check_fitness[i] > bestFit)
                {
                    bestFit = result_check_fitness[i];
                }
            }

            // Copy the best route to host
            // int* h_route = new int[CITY_COUNT];
            // CUDA_CHECK(cudaMemcpy(h_route, d_population + bestIdx * CITY_COUNT, CITY_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

            // Compute its distance on the CPU (assuming you have a distanceMatrix on CPU as h_distMat)
            double bestDistance = 1.0 / bestFit;

            cout << "Generation " << gen << ": Best Distance = " << bestDistance << endl;

            // delete[] h_route;
        }
    }

    // After final generation:
    // Compute fitness once more to ensure updated:
    computeFitnessKernel<<<blocks, threads>>>(d_population, d_distanceMatix, d_fitness, POPULATION_SIZE, CITY_COUNT);
    cudaDeviceSynchronize();

    // Copy fitness back to CPU
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time usage: " << elapsed.count() << " s" << endl;

    double bestDistance = 1.0 / bestFit;
    cout << "Best distance: " << bestDistance << endl;
    cout << "Best path:" << endl;
    for (int i = 0; i < CITY_COUNT; i++)
        cout << h_solution[i] << " ";
    cout << "\n";

    delete[] h_solution;
    delete[] h_fitness;
    delete[] cities;
    delete[] h_distMat;

    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_fitness);
    cudaFree(d_selectedIndices);
    cudaFree(d_distanceMatix);
    cudaFree(d_states);

    return 0;
}
