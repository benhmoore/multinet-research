import torch.nn as nn
import networkx as nx


def generate_adjacency_matrix(model: nn.Module):
    """Generates the adjacency matrix of a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to generate the adjacency matrix of.

    Returns:
        np.array: Adjacency matrix of the PyTorch model.
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
