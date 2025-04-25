
import os
import yaml
import time
import psutil
import itertools

def held_karp(distance_matrix):
    # Get the current process to monitor
    process = psutil.Process(os.getpid())
    
    # Start CPU and memory monitoring
    start_time = time.time()
    start_cpu_times = process.cpu_times()
    start_memory = process.memory_info().rss
    
    # Original algorithm
    n = len(distance_matrix)
    C = {}
    # Initialize the distances for subsets of size 1
    for k in range(1, n):
        C[(1 << k, k)] = (distance_matrix[0][k], 0)
    
    # Iterate through subsets of increasing length
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    res.append((C[(prev_bits, m)][0] + distance_matrix[m][k], m))
                C[(bits, k)] = min(res)
    
    # Find the optimal cost and path
    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distance_matrix[k][0], k))
    opt, parent = min(res)
    
    # Backtrack to find the full path
    path = [0]
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits
    path.append(0)
    
    # End measurement
    end_time = time.time()
    end_cpu_times = process.cpu_times()
    end_memory = process.memory_info().rss
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    
    # Calculate CPU usage
    user_cpu_time = end_cpu_times.user - start_cpu_times.user
    system_cpu_time = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = user_cpu_time + system_cpu_time
    
    # CPU percentage is the fraction of CPU time out of wall time, multiplied by 100
    # This gives the average CPU usage over the execution period
    cpu_percent = (total_cpu_time / elapsed_time) * 100
    
    # Memory usage in bytes
    memory_used = end_memory - start_memory
    
    # Return original result plus performance metrics
    return {
        "result": (opt, path),
        "elapsed_time_seconds": elapsed_time,
        "cpu_percent": cpu_percent,
        "memory_used_bytes": memory_used
    }
