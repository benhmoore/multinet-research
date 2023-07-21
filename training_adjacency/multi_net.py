import pymnet
import torch.nn as nn


def generate_supra_adjacency_matrix(model, model_name="Default"):
    """Converts a PyTorch model to a multilayer network and returns the supra-adjacency matrix.

    Args:
        model (nn.Module): PyTorch model to convert to a multilayer network.
        model_name (str, optional): Name of the model. Defaults to "Default".

    Raises:
        ValueError: If the first module of the model is not an instance of nn.Linear.

    Returns:
        np.array: Supra-adjacency matrix of the multilayer network.

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
