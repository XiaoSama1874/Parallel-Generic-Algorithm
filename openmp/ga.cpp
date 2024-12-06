#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

const int CITY_COUNT = 100;       // 城市数量
const int POPULATION_SIZE = 1000; // 种群规模
const int GENERATIONS = 1000;     // 迭代次数
const double MUTATION_RATE = 0.1; // 变异率
const double MUTATION_PROB = 0.5; // 变异概率
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
    #pragma omp parallel for
    for (int i = 0; i < CITY_COUNT; ++i) {
        cities[i] = {city_coord_dist(gen), city_coord_dist(gen)};
    }
    return cities;
}

// 预计算城市之间的距离并存储在二维数组中
double** calculateDistanceMatrix(const City* cities) {
    double** distanceMatrix = new double*[CITY_COUNT];
    #pragma omp parallel for
    for (int i = 0; i < CITY_COUNT; ++i) {
        distanceMatrix[i] = new double[CITY_COUNT];
        for (int j = 0; j < CITY_COUNT; ++j) {
            if (i == j) {
                distanceMatrix[i][j] = 0.0;
            } else {
                double distance = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
                distanceMatrix[i][j] = distance;
            }
        }
    }
    return distanceMatrix;
}

// 计算路径的总距离
double calculatePathDistance(const int* path, double** distanceMatrix) {
    double totalDistance = 0.0;
    #pragma omp parallel for reduction(+:totalDistance)
    for (int i = 0; i < CITY_COUNT - 1; ++i) {
        totalDistance += distanceMatrix[path[i]][path[i + 1]];
    }
    totalDistance += distanceMatrix[path[CITY_COUNT - 1]][path[0]]; // 返回到起点
    return totalDistance;
}

// 生成随机路径
int* randomPath(int* templatePath) {
    int* path = new int[CITY_COUNT];
    copy(templatePath, templatePath + CITY_COUNT, path); // 复制模板路径
    shuffle(path, path + CITY_COUNT, gen);               // 随机打乱路径
    return path;
}

// 初始化种群
int** initializePopulation(int* templatePath) {
    int** population = new int*[POPULATION_SIZE];
    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i] = randomPath(templatePath);
    }
    return population;
}

// 选择操作（轮盘赌选择）
int* selectParent(int** population, double* fitness, double totalFitness) {
    double r = dis(gen) * totalFitness;
    double cumulativeFitness = 0.0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        cumulativeFitness += fitness[i];
        if (cumulativeFitness >= r) {
            return population[i];
        }
    }
    return population[POPULATION_SIZE - 1];
}

// 交叉操作（部分匹配交叉，PMX）
int* crossover(const int* parent1, const int* parent2) {
    int* offspring = new int[CITY_COUNT];
    fill(offspring, offspring + CITY_COUNT, -1);

    int start = city_dist(gen);
    int end = city_dist(gen);
    if (start > end) swap(start, end);

    // 保留区间[start, end]的基因
    #pragma omp parallel for
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
    return offspring;
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
int* geneticAlgorithm(City* cities, double** distanceMatrix) {
    int* templatePath = new int[CITY_COUNT];
    iota(templatePath, templatePath + CITY_COUNT, 0);

    int** population = initializePopulation(templatePath);
    double* fitness = new double[POPULATION_SIZE];

    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // 计算适应度和总适应度
        double totalFitness = 0.0;
        #pragma omp parallel for reduction(+:totalFitness)
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            fitness[i] = 1.0 / calculatePathDistance(population[i], distanceMatrix);
            totalFitness += fitness[i];
        }

        // 精英保留
        int eliteCount = ELITISM_THRESHOLD * POPULATION_SIZE;
        vector<pair<double, int>> fitnessWithIndex(POPULATION_SIZE);
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            fitnessWithIndex[i] = {fitness[i], i};
        }
        std::partial_sort(
            fitnessWithIndex.begin(),
            fitnessWithIndex.begin() + eliteCount,
            fitnessWithIndex.end(),
            std::greater<>()
        );

        int** newPopulation = new int*[POPULATION_SIZE];
        #pragma omp parallel for
        for (int i = 0; i < eliteCount; ++i) {
            int idx = fitnessWithIndex[i].second;
            newPopulation[i] = new int[CITY_COUNT];
            copy(population[idx], population[idx] + CITY_COUNT, newPopulation[i]);
        }

        // 生成新种群
        #pragma omp parallel for
        for (int i = eliteCount; i < POPULATION_SIZE; ++i) {
            int* parent1 = selectParent(population, fitness, totalFitness);
            int* parent2 = selectParent(population, fitness, totalFitness);
            int* offspring = crossover(parent1, parent2);
            mutate(offspring,generation);
            newPopulation[i] = offspring;
        }

        // 释放旧种群
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            delete[] population[i];
        }
        delete[] population;

        population = newPopulation;

        // 打印当前代的最优距离
        if (PRINT_EACH_ITERATION) {
            double bestDistance = numeric_limits<double>::max();
            #pragma omp parallel for reduction(min:bestDistance)
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                double distance = calculatePathDistance(population[i], distanceMatrix);
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
        double distance = calculatePathDistance(population[i], distanceMatrix);
        if (distance < bestDistance) {
            bestDistance = distance;
            if (bestPath) delete[] bestPath;
            bestPath = new int[CITY_COUNT];
            copy(population[i], population[i] + CITY_COUNT, bestPath);
        }
    }

    // 清理内存
    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        delete[] population[i];
    }
    delete[] population;
    delete[] fitness;
    delete[] templatePath;

    return bestPath;
}

int main() {
    City* cities = initializeCities();
    double** distanceMatrix = calculateDistanceMatrix(cities);

    high_resolution_clock::time_point start = high_resolution_clock::now();
    int* bestPath = geneticAlgorithm(cities, distanceMatrix);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout <<"Time usage:" << duration_sec.count() << endl;

    cout << "最优路径:" << endl;
    for (int i = 0; i < CITY_COUNT; ++i) {
        cout << bestPath[i] << " ";
    }
    cout << endl;

    delete[] bestPath;
    #pragma omp parallel for
    for (int i = 0; i < CITY_COUNT; ++i) {
        delete[] distanceMatrix[i];
    }
    delete[] distanceMatrix;
    delete[] cities;

    return 0;
}
