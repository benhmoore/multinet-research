import pymnet
import torch.nn as nn

def model_to_pymnet_plot(model, model_name="Default"):
    # Create a multilayer network
    net = pymnet.MultilayerNetwork(aspects=1)

    # Initialize a list to store previous layer's neurons
    previous_layer_neurons = [f"input_{i}" for i in range(3)]  # Assuming 3 input neurons
    previous_layer = 'input'

    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            weight_matrix = module.weight.detach().numpy()
            current_layer = f"layer_{idx}"
            current_layer_neurons = [f"{current_layer}_neuron_{i}" for i in range(module.out_features)]

            # Add nodes for the current layer
            for neuron in current_layer_neurons:
                net.add_node(neuron, layer=current_layer)

            # Connect nodes from the previous layer to the current layer
            for i, neuron1 in enumerate(previous_layer_neurons):
                for j, neuron2 in enumerate(current_layer_neurons):
                    net[neuron1, previous_layer][neuron2, current_layer] = weight_matrix[j, i]

            previous_layer_neurons = current_layer_neurons
            previous_layer = current_layer

    # Get the supra-adjacency matrix
    matrix, _ = pymnet.supra_adjacency_matrix(net, includeCouplings=True)

    return matrix
