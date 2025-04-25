# ## Note: Python 3.11.x is required

# !pip install PyYAML numpy
# !pip install qiskit-optimization

import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qiskit_optimization.applications import Tsp

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

number_of_nodes_classical = config['number_of_nodes_classical']
number_of_nodes_classical

def generate_tsp(n):
    tsp = Tsp.create_random_instance(n, seed=123)
    adj_matrix = nx.to_numpy_array(tsp.graph)
    # print("distance\n", adj_matrix)

    colors = ["r" for node in tsp.graph.nodes]
    pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
    # draw_graph(tsp.graph, colors, pos)
    return tsp,adj_matrix, colors, pos


def visualize_tsp_solution(distance_matrix, optimal_path=None):
    distance_matrix = np.array(distance_matrix)
    n = len(distance_matrix)
    
    # Create a complete graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges with weights
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-loops
                G.add_edge(i, j, weight=distance_matrix[i][j])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, 
                          node_color='lightblue', 
                          edgecolors='black',
                          ax=ax)
    
    # Set node labels
    city_labels = {i: f"City {i}" for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=city_labels, font_size=12, font_weight='bold', ax=ax)
    
    # Draw all edges (faded)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, ax=ax)
    
    # Get edge weights for labels (only for the optimal path)
    edge_labels = {}
    
    # If optimal path is provided
    if optimal_path:
        # Calculate total distance
        total_distance = 0
        path_edges = []
        edge_weights = []
        
        # Create the path edges and calculate total distance
        for i in range(len(optimal_path) - 1):
            u, v = optimal_path[i], optimal_path[i+1]
            path_edges.append((u, v))
            distance = distance_matrix[u][v]
            total_distance += distance
            edge_weights.append(distance)
            edge_labels[(u, v)] = f"{distance}"
        
        # Draw the optimal path with directional arrows
        nx.draw_networkx_edges(G, pos, 
                              edgelist=path_edges,
                              width=3.0, 
                              edge_color='red',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax)
        
        # Draw edge labels for the optimal path
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                    font_size=11, font_color='red', 
                                    font_weight='bold', ax=ax)
        
        # Add order indicators along the path (numbers showing the sequence)
        for i, node in enumerate(optimal_path[:-1]):  # Skip the last one as it's a repeat of first
            x, y = pos[node]
            plt.annotate(f'Stop {i+1}', xy=(x, y), xytext=(x, y-0.1),
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
                        ha='center', fontsize=10)
        
        # Add total distance information
        plt.figtext(0.5, 0.01, f"Total distance: {total_distance} units", 
                   ha="center", fontsize=14, bbox={"facecolor":"lightgreen", "alpha":0.5, "pad":5})
        
        # Add a legend
        red_line = mpatches.Patch(color='red', label=f'Optimal Route (Distance: {total_distance})')
        plt.legend(handles=[red_line], loc='upper right', fontsize=12)
        
        # Set title with route information
        route_str = " → ".join([f"{city_labels[city]}" for city in optimal_path])
        plt.title(f"Optimal TSP Route:\n{route_str}", fontsize=14, pad=20)
    else:
        # If no path is provided, show all weights
        edge_labels = {(i, j): f"{distance_matrix[i][j]}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)
        plt.title("TSP Network (No Optimal Route Provided)", fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax
