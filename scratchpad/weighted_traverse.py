import torch
import networkx as nx
import torchvision.models as models
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()


def traverse_model(module, parent=None):
    for name, child in module.named_children():
        # Create a node for the child
        node_name = f"{name}_{id(child)}"
        G.add_node(node_name)

        # Create an edge from the parent to the child with weight as mean of absolute parameters
        if parent is not None:
            if len(list(child.parameters())) > 0:  # Check if the module has parameters
                params = torch.cat(
                    [x.view(-1) for x in child.parameters()]
                )  # Concat all parameters into a 1D tensor
                edge_weight = torch.mean(
                    torch.abs(params)
                ).item()  # Compute mean of absolute values
            else:
                edge_weight = 0  # If the module has no parameters, set the weight to 0

            G.add_edge(parent, node_name, weight=edge_weight)

        # Recursively traverse the child's children
        traverse_model(child, parent=node_name)


# Load the trained model
torch_model = models.resnet18(pretrained=True)  # Load a pretrained model

# Traverse the model
traverse_model(torch_model)

# Get the adjacency matrix
adjacency_matrix = nx.adjacency_matrix(G)

# Convert the adjacency matrix to a dense numpy array
adjacency_array = adjacency_matrix.toarray()

# Plot the adjacency matrix
plt.imshow(adjacency_array, cmap="Greys", interpolation="none")
plt.colorbar(label="Edge Weight")
plt.title("Adjacency Matrix")
plt.show()
