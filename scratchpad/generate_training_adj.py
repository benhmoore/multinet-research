import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Image


from sine_approximator import SineApproximator


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


# create directory for frames
if not os.path.exists("frames"):
    os.makedirs("frames")

# Instantiate the model
model = SineApproximator()

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate input data
x = torch.rand(100, 3) * 6.28 - 3.14
# Generate target data
mu = torch.zeros(3)
sigma = torch.ones(3)

# Complex 3d sine function
y = (
    torch.sin(x).sum(dim=1, keepdim=True)
    + torch.sin(2 * x).sum(dim=1, keepdim=True)
    + torch.sin(0.5 * x).sum(dim=1, keepdim=True)
)

# Add noise
noise = torch.randn(y.size()) * 0.5  # Gaussian noise with mean=0, std=0.1
y += noise

# Training loop
images = []
for epoch in range(2500):
    optimizer.zero_grad()  # zero the gradient buffers

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()  # Does the update

    if epoch % 50 == 0:
        adjacency_array = create_graph(model)

        plt.figure(figsize=(4, 4))
        plt.imshow(adjacency_array, cmap="gray", interpolation="none")
        plt.clim(0, 0.2)  # consistent scale for each frame
        plt.colorbar(label="Edge Weight")
        plt.title(f"Epoch {epoch}; Loss {loss.item():.5f}")
        plt.savefig(f"frames/frame_{epoch}.png")
        plt.close()

        images.append(imageio.imread(f"frames/frame_{epoch}.png"))


imageio.mimsave("frames/training.gif", images, loop=15, duration=50)

# Delete the individual frames
for filename in os.listdir("frames"):
    if filename.endswith(".png"):
        os.remove(f"frames/{filename}")
