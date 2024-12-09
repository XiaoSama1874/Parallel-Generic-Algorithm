#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J sequential
#SBATCH -c 2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 20


g++ ga_sequential.cpp -Wall -O3 -std=c++17 -o ga_sequential

# Define population sizes to test
POPULATION_SIZES=(1000 2000 3000 5000 10000 15000 20000)
# POPULATION_SIZES=(100)


# Loop through each population size
for population_size in "${POPULATION_SIZES[@]}"; do
    # Echo the current population size
    echo "$population_size"
    # Execute the command with the current population size
    ./ga_sequential 100 $population_size 500 false
done