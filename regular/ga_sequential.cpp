#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <vector>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int CITY_COUNT = 100;       // city scale
int POPULATION_SIZE = 1000; // population scale
int GENERATIONS = 1000;     // iteration times
const double MUTATION_RATE = 0.2; // mutation rate
const double ELITISM_THRESHOLD = 0.2; // elite
bool PRINT_EACH_ITERATION = true; // print each iteration's best distance
const int TOURNAMENT_SIZE = 5; // For tournament selection

// Global random engine
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

struct City {
    int x, y;
};


City* initializeCities() {
    City* cities = new City[CITY_COUNT];
    uniform_int_distribution<int> coordDist(0, 100);
    for (int i = 0; i < CITY_COUNT; ++i) {
        cities[i] = {coordDist(gen), coordDist(gen)};
    }
    return cities;
}

double** calculateDistanceMatrix(const City* cities) {
    double** distanceMatrix = new double*[CITY_COUNT];
    for (int i = 0; i < CITY_COUNT; ++i) {
        distanceMatrix[i] = new double[CITY_COUNT];
        for (int j = 0; j < CITY_COUNT; ++j) {
            if (i == j) {
                distanceMatrix[i][j] = 0.0;
            } else {
                double dx = cities[i].x - cities[j].x;
                double dy = cities[i].y - cities[j].y;
                double distance = sqrt(dx * dx + dy * dy);
                distanceMatrix[i][j] = distance;
            }
        }
    }
    return distanceMatrix;
}

double calculatePathDistance(const int* path, double** distanceMatrix) {
    double totalDistance = 0.0;
    for (int i = 0; i < CITY_COUNT - 1; ++i) {
        totalDistance += distanceMatrix[path[i]][path[i + 1]];
    }
    totalDistance += distanceMatrix[path[CITY_COUNT - 1]][path[0]]; 
    return totalDistance;
}

int* randomPath(const int* templatePath) {
    int* path = new int[CITY_COUNT];
    copy(templatePath, templatePath + CITY_COUNT, path);
    shuffle(path, path + CITY_COUNT, gen);
    return path;
}

int** initializePopulation(int* templatePath) {
    int** population = new int*[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i] = randomPath(templatePath);
    }
    return population;
}

// Tournament selection
int* tournamentSelection(int** population, double* fitness) {
    uniform_int_distribution<int> popDist(0, POPULATION_SIZE - 1);
    double bestFit = -1.0;
    int* bestIndividual = nullptr;
    for (int i = 0; i < TOURNAMENT_SIZE; i++) {
        int cand = popDist(gen);
        double f = fitness[cand];
        if (f > bestFit) {
            bestFit = f;
            bestIndividual = population[cand];
        }
    }
    return bestIndividual;
}

// Order Crossover (OX)
void orderCrossover(const int* parent1, const int* parent2, int* child1, int* child2) {
    fill(child1, child1 + CITY_COUNT, -1);
    fill(child2, child2 + CITY_COUNT, -1);


    uniform_int_distribution<int> cityDist(0, CITY_COUNT - 1);
    int start = cityDist(gen);
    int end = cityDist(gen);
    if (start > end) swap(start, end);


    for (int i = start; i <= end; i++) {
        child1[i] = parent1[i];
    }

    {
        int pos = (end + 1) % CITY_COUNT;
        for (int i = 0; i < CITY_COUNT; i++) {
            int gene = parent2[i];
            if (find(child1, child1 + CITY_COUNT, gene) == child1 + CITY_COUNT) {
                child1[pos] = gene;
                pos = (pos + 1) % CITY_COUNT;
            }
        }
    }

    for (int i = start; i <= end; i++) {
        child2[i] = parent2[i];
    }

    {
        int pos = (end + 1) % CITY_COUNT;
        for (int i = 0; i < CITY_COUNT; i++) {
            int gene = parent1[i];
            if (find(child2, child2 + CITY_COUNT, gene) == child2 + CITY_COUNT) {
                child2[pos] = gene;
                pos = (pos + 1) % CITY_COUNT;
            }
        }
    }
}

void mutate(int* individual) {
    if (dis(gen) < MUTATION_RATE) {
        uniform_int_distribution<int> cityDist(0, CITY_COUNT - 1);
        int i = cityDist(gen);
        int j = cityDist(gen);
        swap(individual[i], individual[j]);
    }
}

void geneticAlgorithm(City* cities, double** distanceMatrix) {
    int* templatePath = new int[CITY_COUNT];
    iota(templatePath, templatePath + CITY_COUNT, 0);

    int** population = initializePopulation(templatePath);

    int** newPopulation = new int*[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        newPopulation[i] = new int[CITY_COUNT];
    }

    double* fitness = new double[POPULATION_SIZE];

    for (int generation = 0; generation < GENERATIONS; ++generation) {
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            double dist = calculatePathDistance(population[i], distanceMatrix);
            fitness[i] = 1.0 / dist;
        }

        int eliteCount = (int)(ELITISM_THRESHOLD * POPULATION_SIZE);
        vector<pair<double, int>> fitnessWithIndex;
        fitnessWithIndex.reserve(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            fitnessWithIndex.push_back({fitness[i], i});
        }
        sort(fitnessWithIndex.begin(), fitnessWithIndex.end(), greater<>());

        for (int i = 0; i < eliteCount; ++i) {
            int idx = fitnessWithIndex[i].second;
            copy(population[idx], population[idx] + CITY_COUNT, newPopulation[i]);
        }

        int nonElite = POPULATION_SIZE - eliteCount;
        int halfNonElite = nonElite / 2;
        int writeIdx = eliteCount;

        for (int pair_i = 0; pair_i < halfNonElite; pair_i++) {
            int* parent1 = tournamentSelection(population, fitness);
            int* parent2 = tournamentSelection(population, fitness);
            orderCrossover(parent1, parent2, newPopulation[writeIdx], newPopulation[writeIdx + 1]);
            mutate(newPopulation[writeIdx]);
            mutate(newPopulation[writeIdx + 1]);
            writeIdx += 2;
        }

        int** temp = population;
        population = newPopulation;
        newPopulation = temp;

        if (PRINT_EACH_ITERATION && (generation % 50 == 0)) {
            double bestDistance = numeric_limits<double>::max();
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                double distance = calculatePathDistance(population[i], distanceMatrix);
                if (distance < bestDistance) {
                    bestDistance = distance;
                }
            }
            cout << "Generation " << generation << ": Best Distance = " << bestDistance << endl;
        }
    }

    double bestDistance = numeric_limits<double>::max();
    int bestDistanceIndex = -1;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        double distance = calculatePathDistance(population[i], distanceMatrix);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestDistanceIndex = i;
        }
    }

    int* route = new int[CITY_COUNT];
    if (bestDistanceIndex != -1) {
        copy(population[bestDistanceIndex], population[bestDistanceIndex] + CITY_COUNT, route);
    }

    cout << "Best path:" << endl;
    for (int i = 0; i < CITY_COUNT; ++i) {
        cout << route[i] << " ";
    }
    cout << endl;

    // Cleanup memory
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        delete[] population[i];
        delete[] newPopulation[i];
    }
    delete[] population;
    delete[] newPopulation;
    delete[] fitness;
    delete[] templatePath;
    delete[] route;
}

int main(int argc, char* argv[]) {
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

    City* cities = initializeCities();
    double** distanceMatrix = calculateDistanceMatrix(cities);

    high_resolution_clock::time_point start = high_resolution_clock::now();

    geneticAlgorithm(cities, distanceMatrix);

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout <<"Time usage:" << duration_sec.count() << " ms" << endl;

    for (int i = 0; i < CITY_COUNT; ++i) {
        delete[] distanceMatrix[i];
    }
    delete[] distanceMatrix;
    delete[] cities;

    return 0;
}
