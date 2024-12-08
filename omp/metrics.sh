#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J omp
#SBATCH -c 2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 10
#SBATCH --cpus-per-task=8


g++ ga_omp.cpp -Wall -O3 -std=c++17 -fopenmp -o ga_omp

# Define population sizes to test
POPULATION_SIZES=(100 500 1000 2000 3000 5000)
# POPULATION_SIZES=(100)


# Loop through each population size
for population_size in "${POPULATION_SIZES[@]}"; do
    # Echo the current population size
    echo "$population_size"
    # Execute the command with the current population size
    ./ga_omp 500 $population_size 1000 false
done