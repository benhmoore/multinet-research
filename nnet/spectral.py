import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


# Initialize a Barabasi-Albert (BA) graph with 100 nodes
n = 100  # Number of nodes
m = 2  # Number of edges to attach from a new node to existing nodes
ba_graph = nx.barabasi_albert_graph(n, m)

# Perform eigendecomposition on the adjacency matrix
adj_matrix = nx.adjacency_matrix(ba_graph)
adj_array = adj_matrix.toarray()
eigenvalues = np.linalg.eigvalsh(adj_array)

# Calculate the spectral density
density, bins, _ = plt.hist(eigenvalues, bins="auto", density=True)

# Plot the spectral density
plt.xlabel("Eigenvalue")
plt.ylabel("Density")
plt.title("Spectral Density of BA Graph Eigenvalues")
plt.show()
