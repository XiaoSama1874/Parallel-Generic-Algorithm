#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH -J ga_cuda
#SBATCH -o %x.out -e %x.err
#SBATCH --time=0-00:10:00

# sbatch task1.sh

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc ga_cuda.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o ga_cuda

# Define population sizes to test
POPULATION_SIZES=(1000 2000 3000 5000 10000 15000 20000)
# POPULATION_SIZES=(100)


# Loop through each population size
for population_size in "${POPULATION_SIZES[@]}"; do
    # Echo the current population size
    echo "$population_size"
    
    # Execute the command with the current population size
    ./ga_cuda 100 $population_size 200 false
done