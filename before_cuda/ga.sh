#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH -J ga_slurm
#SBATCH -o ga_slurm.out
#SBATCH -e ga_slurm.err

#SBATCH --time=0-00:02:00

# sbatch ga.sh

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc ga.cu ga_helper.cu ga_helper2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o ga
./ga

# squeue -j 601632
