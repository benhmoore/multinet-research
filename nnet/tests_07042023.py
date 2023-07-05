import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.linalg import eigvals

import matplotlib.pyplot as plt

# Generate a graph representing a simple neural network:
#  [input] -> [Node01] -> [Node02]
#          -> [Node11] -> [Node12] -> [output]
#          -> [Node21] ->

nnet = nx.Graph()

# Define neurons
nnet.add_nodes_from(["input"])
nnet.add_nodes_from(["node01", "node11", "node21"])  # Hidden layer 1
nnet.add_nodes_from(["node02", "node12"])  # Hidden layer 2
nnet.add_nodes_from(["output"])

# Define edges
# input
nnet.add_edges_from([("input", "node01"), ("input", "node11"), ("input", "node21")])

# node01
nnet.add_edges_from([("node01", "node02"), ("node01", "node12")])

# node11
nnet.add_edges_from([("node11", "node02"), ("node11", "node12")])

# node21
nnet.add_edges_from([("node21", "node02"), ("node21", "node12")])

# node02
nnet.add_edges_from([("node02", "output")])

# node12
nnet.add_edges_from([("node12", "output")])


adj_matrix = nx.adjacency_matrix(nnet)
adj_array = adj_matrix.toarray()
eigenvalues = np.linalg.eigvalsh(adj_array)
print(adj_array, eigenvalues)

# Calculate the spectral density
density, bins, _ = plt.hist(eigenvalues, bins="auto", density=True)


# Plot the spectral density
plt.xlabel("Eigenvalue")
plt.ylabel("Density")
plt.title("Spectral Density of BA Graph Eigenvalues")
plt.show()

nx.draw(nnet, with_labels=True)
plt.show()
