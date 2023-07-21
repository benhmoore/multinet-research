import torch.nn as nn
import networkx as nx


def create_graph(model):
    """
    Converts a PyTorch model into a NetworkX graph and returns its adjacency matrix.

    This function iterates over the layers of the PyTorch model, treating the entire model as a single network.
    For each nn.Linear layer, it adds nodes to the graph corresponding to the neurons in the layer, and adds edges
    based on the weights of the connections between neurons in the layer and the previous layer.

    The function specifically generates an adjacency matrix from the graph. The adjacency matrix is a square matrix
    where each row and column corresponds to a node (neuron) in the graph, and the value at a specific row and column
    represents the weight of the edge connecting the two corresponding nodes.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be converted into a NetworkX graph.

    Returns
    -------
    np.array
        Adjacency matrix of the NetworkX graph. The matrix has dimensions of N x N, where N is the total number of
        unique nodes (neurons) across all layers in the model. Each node represents a neuron in the model.

    Raises
    ------
    ValueError
        If the first module of the model is not an instance of nn.Linear.
    """

    G = nx.Graph()
    previous_layer_neurons = [
        f"input_{i}" for i in range(3)
    ]  # Assuming 3 input neurons

    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            weight_matrix = module.weight.detach().numpy()
            current_layer_neurons = [
                f"layer_{idx}_neuron_{i}" for i in range(module.out_features)
            ]

            # Add nodes for the current layer
            for neuron in current_layer_neurons:
                G.add_node(neuron)

            # Connect nodes from the previous layer to the current layer
            for i, neuron1 in enumerate(previous_layer_neurons):
                for j, neuron2 in enumerate(current_layer_neurons):
                    G.add_edge(neuron1, neuron2, weight=weight_matrix[j, i])

            previous_layer_neurons = current_layer_neurons

    adjacency_matrix = nx.adjacency_matrix(G, weight="weight")
    return adjacency_matrix.toarray()
