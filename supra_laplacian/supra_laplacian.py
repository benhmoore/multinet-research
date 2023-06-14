import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import eigvals


def calculate_supra_laplacian(
    network_layers, interlayer_edges: list, interlayer_weight: float = 1.0
):
    """Calculate the supra-Laplacian matrix of a multiplex network.

    Args:
        network_layers (list[networkx.Graph]): List of network layers.
        interlayer_edges (list[tuple]): List of interlayer edges.
        interlayer_weight (float, optional): Weight of interlayer edges. Defaults to 1.0.

    Returns:
        numpy.ndarray: Supra-Laplacian matrix of the multiplex network.
    """
    n = len(network_layers[0].nodes)  # number of nodes in each layer
    m = len(network_layers)  # number of layers

    # Create aggregated adjacency matrix
    adj_matrices = [nx.adjacency_matrix(G).toarray() for G in network_layers]
    aggregated_adj_matrix = block_diag(*adj_matrices)

    # Create supra-Laplacian matrix from aggregated adjacency matrix
    # The supra-Laplacian matrix is a block diagonal matrix where each
    # block corresponds to a layer in the multiplex network.
    # The size of each block is n x n, where n is the number of
    # nodes in each layer.
    supra_L = (
        nx.laplacian_matrix(nx.from_numpy_array(aggregated_adj_matrix))
        .toarray()
        .astype(float)
    )

    # Update supra-Laplacian to account for the interlayer edges
    for edge in interlayer_edges:
        i, j, layer_i, layer_j = edge

        # Add the interlayer weight to each interlayer edge in the
        # supra-Laplacian matrix using list slicing.
        supra_L[
            layer_i * n : (layer_i + 1) * n, layer_j * n : (layer_j + 1) * n
        ] += interlayer_weight
        supra_L[
            layer_j * n : (layer_j + 1) * n, layer_i * n : (layer_i + 1) * n
        ] += interlayer_weight

    return supra_L
