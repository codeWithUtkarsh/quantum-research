
import os
import time
import psutil
import pickle
import tracemalloc
import matplotlib.pyplot as plt
from held_karp_classical import held_karp
from utils import generate_tsp, visualize_tsp_solution, number_of_nodes_classical

mem_cpu_stat = {}
def run_test_classical(num_vertices, edge_prob=0.5):

    _, adj_matrix, _, _ = generate_tsp(num_vertices)
    # fig1, ax1 = visualize_tsp_network(adj_matrix)
    
    plt.show()
    
    start_time = time.time()
    tracemalloc.start()
    process = psutil.Process()
    mem_before = process.memory_info().rss

    # opt, path = held_karp(adj_matrix)
    result = held_karp(adj_matrix)
    opt, path = result['result']
    print("="*50)
    print(f"REQUIREMENT FOR Nodes :: {num_vertices}")
    print(f"CPU usage: {result['cpu_percent']:.2f}%")
    print(f"Memory used: {result['memory_used_bytes'] / (1024 * 1024):.2f} MB")
    
    print("="*50)
    print("Network Visualization with highlighted optimal path")
    
    mem_cpu_stat[num_vertices] = { "cpu": result['cpu_percent'], "mem": result['memory_used_bytes'], "time": result['elapsed_time_seconds'] }
    
    # fig2, ax2 = visualize_tsp_solution(adj_matrix, path)
    
    end_time = time.time()
    mem_after = process.memory_info().rss
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_taken = end_time - start_time
    memory_used = (mem_after - mem_before) / 10**6  # in MB
    peak_memory_used = peak / 10**6  # in MB

    print(f'time_taken for Nodes {num_vertices} :: {time_taken}')
    return time_taken


time_classical = []
file_path_classical = 'tsp/cpu_classical_stat.pkl'
force_rerun = True

# Check if the file already exists
if os.path.exists(file_path_classical) and not force_rerun:
    print(f"File {file_path_classical} already exists. It will not be overwritten.")

    with open(file_path_classical, 'rb') as f:
        loaded_dict = pickle.load(f)
        
    time_classical_dict = loaded_dict
else:

    for node in number_of_nodes_classical:
        time_taken = run_test_classical(node)
        time_classical.append(time_taken)
        
    time_classical_dict = {}
    for i, node in enumerate(number_of_nodes_classical):
        time_classical_dict[node] = time_classical[i]

    # Saving the structure
    with open(file_path_classical, 'wb') as f:
        pickle.dump(time_classical_dict, f)
    print(f"Dictionary has been saved to {file_path_classical} successfully.")


