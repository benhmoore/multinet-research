import pymnet
from torch import nn


def model_to_pymnet_plot(model, model_name="Default"):
    # Create a multilayer network
    net = pymnet.MultilayerNetwork(aspects=1)

    def traverse_model(module, parent=None, layer=None):
        for name, child in module.named_children():
            # Create a node for the child
            node_name = f"{name}_{id(child)}"
            net.add_node(node_name, layer=layer)

            # Create an edge from the parent to the child
            if parent is not None:
                if isinstance(child, nn.Linear):
                    weight = child.weight.detach().numpy().mean()  # Get the mean weight
                else:
                    weight = 1
                net[parent, layer][node_name, layer] = weight

            # Recursively traverse the child's children
            traverse_model(child, parent=node_name, layer=layer)

    # Traverse the model
    for i, (name, module) in enumerate(model.named_children()):
        traverse_model(module, layer=name)

    # Get the supra-adjacency matrix
    matrix, _ = pymnet.supra_adjacency_matrix(net, includeCouplings=True)

    return matrix
