import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Image


def create_graph(module, parent=None):
    G = nx.Graph()
    traverse_model(module, G, parent)
    adjacency_matrix = nx.adjacency_matrix(G)
    return adjacency_matrix.toarray()


def traverse_model(module, G, parent=None):
    previous_node = parent
    for name, child in module.named_children():
        node_name = f"{name}_{id(child)}"
        G.add_node(node_name)

        if previous_node is not None:
            edge_weight = 0
            if len(list(child.parameters())) > 0:
                params = torch.cat([x.view(-1) for x in child.parameters()])
                edge_weight = torch.mean(torch.abs(params)).item()

            if G.has_edge(previous_node, node_name):
                G[previous_node][node_name]["weight"] = edge_weight
            else:
                G.add_edge(previous_node, node_name, weight=edge_weight)

        previous_node = node_name


## Define the model
class SineApproximator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(SineApproximator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc7 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc8 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc9 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc10 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()  # ReLU activation function

        # # Define skip/residual connections
        # self.shortcut1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.shortcut2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.shortcut3 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        # identity = x

        x = self.relu(self.fc1(x))
        # x = x + self.shortcut1(identity)  # Add the first shortcut connection

        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = x + self.shortcut2(x)  # Add the second shortcut connection

        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        # identity2 = x  # Save the input to the second half of the network
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        # x = x + self.shortcut3(identity2)  # Add the third shortcut connection

        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))

        return self.fc10(x)


# create directory for frames
if not os.path.exists("frames"):
    os.makedirs("frames")

# Instantiate the model
model = SineApproximator()

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Generate input data
x = torch.rand(100, 3) * 6.28 - 3.14
# Generate target data
mu = torch.zeros(3)
sigma = torch.ones(3)

# Gaussian Function (3D Gaussian Bell Curve)
y = torch.exp(-0.5 * ((x - mu) / sigma) ** 2).prod(dim=1, keepdim=True)

# Add noise
noise = torch.randn(y.size()) * 0.5  # Gaussian noise with mean=0, std=0.1
y += noise

# Training loop
images = []
for epoch in range(2500):
    optimizer.zero_grad()  # zero the gradient buffers

    # Switch between functions every 500 epochs
    if epoch < 500:
        y = torch.exp(-0.5 * ((x - mu) / sigma) ** 2).prod(dim=1, keepdim=True)
    elif epoch < 1000:  # Switch to an inverted Gaussian function
        y = 1 - torch.exp(-0.5 * ((x - mu) / sigma) ** 2).prod(dim=1, keepdim=True)
    elif epoch < 1500:  # Switch to a Gaussian mixture model
        y = (
            torch.exp(-0.5 * ((x - mu) / sigma) ** 2).prod(dim=1, keepdim=True)
            + torch.exp(-0.5 * ((x - mu + 1) / sigma) ** 2).prod(dim=1, keepdim=True)
            + torch.exp(-0.5 * ((x - mu - 1) / sigma) ** 2).prod(dim=1, keepdim=True)
        )
    else:  # Switch to a complex sine function
        y = (
            torch.sin(x).sum(dim=1, keepdim=True)
            + torch.sin(2 * x).sum(dim=1, keepdim=True)
            + torch.sin(0.5 * x).sum(dim=1, keepdim=True)
        )

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
