import matplotlib.pyplot as plt

def read_metrics(file_path):
    """
    Read metrics from a given file.
    The file format is:
    - First line: population size
    - Second line: execution time in ms
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Parse the data as pairs of population size and execution time
    population_sizes = [int(lines[i].strip()) for i in range(0, len(lines), 2)]
    execution_times = [float(lines[i].strip()) for i in range(1, len(lines), 2)]
    return population_sizes, execution_times

# Read data from files
cuda_pop, cuda_times = read_metrics('metrics_cuda_500_1000.txt')
omp_pop, omp_times = read_metrics('metrics_omp_500_1000.txt')
seq_pop, seq_times = read_metrics('metrics_sequential_500_1000.txt')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(cuda_pop, cuda_times, marker='o', label='CUDA')
plt.plot(omp_pop, omp_times, marker='s', label='OpenMP')
plt.plot(seq_pop, seq_times, marker='^', label='Sequential')

# Add titles and labels
plt.title('Execution Time vs Population Size', fontsize=16)
plt.xlabel('Population Size', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
plt.savefig('execution_time_comparison.png', dpi=300)
plt.show()
