import pymnet
import torch.nn as nn


def model_to_pymnet_plot(model, model_name="Default"):
    """
    Converts a PyTorch model into a pymnet multilayer network and returns its supra-adjacency matrix.

    This function iterates through the layers of the model, creates a layer in the multilayer network for each
    nn.Linear layer, and adds nodes to each layer corresponding to the neurons in the model layer. It then
    connects these nodes based on the weights of the model's layers.

    The function specifically generates a supra-adjacency matrix from the multilayer network. The supra-adjacency
    matrix is a block matrix where each block represents the adjacency matrix of a single layer (diagonal blocks)
    and the connections between layers (off-diagonal blocks).

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be converted into a multilayer network.
    model_name : str, optional
        Name of the model, by default "Default".

    Returns
    -------
    np.array
        Supra-adjacency matrix of the multilayer network. The matrix has a dimension of (N*M) x (N*M), where N is
        the total number of unique nodes (or neurons) across all layers, and M is the total number of layers.
        Each node represents a neuron in the model and each layer in the multilayer network corresponds to an
        nn.Linear layer in the model.

    Raises
    ------
    ValueError
        If the first module of the model is not an instance of nn.Linear.
    """

    # Create a multilayer network
    net = pymnet.MultilayerNetwork(aspects=1)

    # Initialize a list to store previous layer's neurons
    previous_layer_neurons = None
    previous_layer = None

    for idx, module in enumerate(list(model.modules())[1:]):
        if isinstance(module, nn.Linear):
            weight_matrix = module.weight.detach().numpy()
            current_layer = f"layer_{idx}"

            # Assuming input layer for idx=0
            if idx == 0:
                # Infer number of input features
                input_features = module.in_features
                previous_layer_neurons = [f"input_{i}" for i in range(input_features)]
                previous_layer = "input"

                # Add nodes for the input layer
                for neuron in previous_layer_neurons:
                    net.add_node(neuron, layer=previous_layer)

            # If previous_layer_neurons is still None, then the first module is not Linear
            if previous_layer_neurons is None:
                raise ValueError(
                    "The first module of the model is not an instance of nn.Linear"
                )

            current_layer_neurons = [
                f"{current_layer}_neuron_{i}" for i in range(module.out_features)
            ]

            # Add nodes for the current layer
            for neuron in current_layer_neurons:
                net.add_node(neuron, layer=current_layer)

            # Connect nodes from the previous layer to the current layer
            for i, neuron1 in enumerate(previous_layer_neurons):
                for j, neuron2 in enumerate(current_layer_neurons):
                    net[neuron1, previous_layer][
                        neuron2, current_layer
                    ] = weight_matrix[j, i]

            previous_layer_neurons = current_layer_neurons
            previous_layer = current_layer

    # Get the supra-adjacency matrix
    matrix, _ = pymnet.supra_adjacency_matrix(net, includeCouplings=True)

    return matrix
