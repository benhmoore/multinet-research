import torch
import networkx as nx


def create_graph(module, parent=None):
    G = nx.Graph()
    traverse_model(module, G, parent)
    adjacency_matrix = nx.adjacency_matrix(G)
    return adjacency_matrix.toarray()


def traverse_model(module, G, parent=None):
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
        traverse_model(child, G, parent=node_name)