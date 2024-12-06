#include "ga_helper.cuh"
#include "ga_helper2.cuh"
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

const int CITY_COUNT = 100;       // 城市数量
const int POPULATION_SIZE = 1000; // 种群规模
const int GENERATIONS = 1000;     // 迭代次数
const double MUTATION_RATE = 0.1; // 变异率
// const double MUTATION_PROB = 0.5; // 变异概率
const double ELITISM_THRESHOLD = 0.2; // 精英保留比例（20%）
const bool PRINT_EACH_ITERATION = false; // 是否打印每次迭代的最优距离

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);
uniform_int_distribution<> city_dist(0, CITY_COUNT-1);
uniform_int_distribution<> city_coord_dist(0, 100);

// 城市结构
struct City {
    int x, y;
};

// 初始化城市位置
City* initializeCities() {
    City* cities = new City[CITY_COUNT];
    for (int i = 0; i < CITY_COUNT; ++i) {
        cities[i] = {city_coord_dist(gen), city_coord_dist(gen)};
    }
    return cities;
}

// 预计算城市之间的距离并存储在二维数组中 cuda
// double* calculateDistanceMatrix(const City* cities) {
//     double* distanceMatrix;
//     distanceMatrix = new double[CITY_COUNT * CITY_COUNT];
//     for (int idx = 0; idx < CITY_COUNT * CITY_COUNT; ++idx ) {
//         int i = idx / CITY_COUNT;
//         int j = idx % CITY_COUNT;
//         double distance = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
//         distanceMatrix[idx] = distance;
//     }
//     return distanceMatrix;
// }

// 计算路径的总距离
// double calculatePathDistance(const int* path, double* distanceMatrix) {
//     double totalDistance = distanceMatrix[path[CITY_COUNT - 1]*CITY_COUNT + path[0]]; // 返回到起点
//     for (int i = 0; i < CITY_COUNT - 1; ++i) {
//         totalDistance += distanceMatrix[path[i]*CITY_COUNT + path[i + 1]];
//     }
//     return totalDistance;
// }

// 初始化种群
int* initializePopulation() {
    int* population = new int[POPULATION_SIZE * CITY_COUNT];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        int* population_i = population + i * CITY_COUNT;
        iota(population_i, population_i + CITY_COUNT, 0); // 模板路径
        shuffle(population_i, population_i + CITY_COUNT, gen);  // 生成随机路径  
    }
    return population;
}

// 选择操作（轮盘赌选择）
int selectParentIdx(double* fitness, double totalFitness) {
    double r = dis(gen) * totalFitness;
    double cumulativeFitness = 0.0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        cumulativeFitness += fitness[i];
        if (cumulativeFitness >= r) {
            return i;
        }
    }
    return POPULATION_SIZE - 1;
}

// 交叉操作（部分匹配交叉，PMX）
void crossover(const int* parent1, const int* parent2, int* offspring) {
    fill(offspring, offspring + CITY_COUNT, -1);

    int start = city_dist(gen);
    int end = city_dist(gen);
    if (start > end) swap(start, end);

    // 保留区间[start, end]的基因
    for (int i = start; i <= end; ++i) {
        offspring[i] = parent1[i];
    }

    // 填充其余部分
    for (int i = 0; i < CITY_COUNT; ++i) {
        if (find(offspring, offspring + CITY_COUNT, parent2[i]) == offspring + CITY_COUNT) {
            for (int j = 0; j < CITY_COUNT; ++j) {
                if (offspring[j] == -1) {
                    offspring[j] = parent2[i];
                    break;
                }
            }
        }
    }
}


// 变异操作
void mutate(int* individual, int generation) {
    if (dis(gen) < MUTATION_RATE) {
        int i = city_dist(gen);
        int j = city_dist(gen);
        swap(individual[i], individual[j]);
    }
}


// 主遗传算法函数
int* geneticAlgorithm(City* cities, double* distanceMatrix, int threadsPerBlock) {
    int* population = initializePopulation();
    double* sumDistances = new double[POPULATION_SIZE];
    double* fitness = new double[POPULATION_SIZE];
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // 计算适应度和总适应度 // cuda
        double totalFitness = 0.0;
        sumDistances = calculateSumDistances(population, distanceMatrix, POPULATION_SIZE, CITY_COUNT, threadsPerBlock);
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            fitness[i] = 1.0 / sumDistances[i];
            totalFitness += fitness[i];
        }

        // 精英保留
        int eliteCount = ELITISM_THRESHOLD * POPULATION_SIZE;
        vector<pair<double, int>> fitnessWithIndex(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            fitnessWithIndex[i] = {fitness[i], i};
        }
        std::partial_sort(
            fitnessWithIndex.begin(),
            fitnessWithIndex.begin() + eliteCount,
            fitnessWithIndex.end(),
            std::greater<>()
        );

        int* newPopulation = new int[POPULATION_SIZE * CITY_COUNT];
        for (int i = 0; i < eliteCount; ++i) {
            int idx = fitnessWithIndex[i].second;
            int* population_idx = population + idx*CITY_COUNT;
            int* newPopulation_i = newPopulation + i*CITY_COUNT;
            copy(population_idx, population_idx + CITY_COUNT, newPopulation_i);
        }

        // 生成新种群 // cuda
        // #pragma omp parallel for
        for (int i = eliteCount; i < POPULATION_SIZE; ++i) {
            int parent1Idx = selectParentIdx(fitness, totalFitness);
            int* parent1 = population + parent1Idx*CITY_COUNT;
            int parent2Idx = selectParentIdx(fitness, totalFitness);
            int* parent2 = population + parent2Idx*CITY_COUNT;
            int* newPopulation_i = newPopulation + i*CITY_COUNT;
            crossover(parent1, parent2, newPopulation_i);
            mutate(newPopulation_i,generation);
        }

        // 释放旧种群
        delete[] population;
        population = newPopulation;

        // 打印当前代的最优距离
        if (PRINT_EACH_ITERATION) {
            double bestDistance = numeric_limits<double>::max();
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                double distance = sumDistances[i];
                if (distance < bestDistance) {
                    bestDistance = distance;
                }
            }
            cout << "Generation " << generation + 1 << ": Best Distance = " << bestDistance << endl;
        }
    }

    // 找到最优解
    double bestDistance = numeric_limits<double>::max();
    int* bestPath = nullptr;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        int* population_i = population + i*CITY_COUNT;
        double distance = sumDistances[i];
        if (distance < bestDistance) {
            bestDistance = distance;
            if (bestPath) delete[] bestPath;
            bestPath = new int[CITY_COUNT];
            copy(population_i, population_i + CITY_COUNT, bestPath);
        }
    }

    // 清理内存
    delete[] population;
    delete[] fitness;

    return bestPath;
}

int main() {
    City* cities = initializeCities();
    int threadsPerBlock = 32;
    int *hCityX, *hCityY;
    hCityX = new int[CITY_COUNT];
    hCityY = new int[CITY_COUNT];
    for (int i = 0; i < CITY_COUNT; ++i) {
        hCityX[i] = cities[i].x;
        hCityY[i] = cities[i].y;
    }
    double* distanceMatrix = calculateDistanceMatrix(hCityX, hCityY, CITY_COUNT, threadsPerBlock);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    int* bestPath = geneticAlgorithm(cities, distanceMatrix, threadsPerBlock);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout <<"Time usage:" << duration_sec.count() << endl;

    cout << "最优路径:" << endl;
    for (int i = 0; i < CITY_COUNT; ++i) {
        cout << bestPath[i] << " ";
    }
    cout << endl;

    delete[] bestPath;
    delete[] distanceMatrix;
    delete[] cities;

    return 0;
}
