import networkx as nx
from supra_laplacian import calculate_supra_laplacian
from scipy.linalg import eigvals

# - - - - - - Demonstration of the supra-Laplacian matrix calculator - - - - - -

# Create two layers, each with three nodes and some edges.
# Layer 1: Friendship connections
G1 = nx.Graph()
G1.add_nodes_from([0, 1, 2])
G1.add_edges_from(
    [(0, 1), (1, 2)]
)  # Node 0 is friends with Node 1, and Node 1 is friends with Node 2

# Layer 2: Professional connections
G2 = nx.Graph()
G2.add_nodes_from([0, 1, 2])
G2.add_edges_from([(0, 2)])  # Node 0 and Node 2 are professionally connected

# Combine the two layers into a list to represent a multiplex network.
Gs = [G1, G2]

# Define the interlayer edges and their weight.
# Format: (node i, node j, layer i, layer j)
interlayer_edges = [(0, 0, 0, 1), (1, 1, 0, 1)]
interlayer_weight = 0.5

# Calculate the supra-Laplacian matrix of the multiplex network.
supra_L = calculate_supra_laplacian(Gs, interlayer_edges, interlayer_weight)

# Get second eigenvalue of supra-Laplacian matrix.
eigenvals = sorted(eigvals(supra_L))


# - - - - - - Print results - - - - - -

print("Layer 1: ", G1.nodes)
print("Layer 2: ", G2.nodes)
print("Interlayer edges: ", interlayer_edges)
print("----------------------------------------")

print("Layer 1 Laplacian matrix: \n", nx.laplacian_matrix(G1).toarray())
print("Layer 2 Laplacian matrix: \n", nx.laplacian_matrix(G2).toarray())

print("Supra-Laplacian matrix: \n", supra_L)
print("Eigenvalues: \n", eigenvals)
print("----------------------------------------")
