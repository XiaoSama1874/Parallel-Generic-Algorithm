#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH -J task1_slurm
#SBATCH -o task1_slurm.out
#SBATCH -e task1_slurm.err

#SBATCH --time=0-00:02:00

# sbatch task1.sh

module load nvidia/cuda/11.8.0
module load gcc/11.3.0


nvcc -Xcompiler "-fopenmp" -ccbin g++ -o combined_cuda_omp cudaAndOpenMP.cu
./combined_cuda_omp 