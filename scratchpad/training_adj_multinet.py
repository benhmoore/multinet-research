import pymnet
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os


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
                net[parent, layer][node_name, layer] = 1

            # Recursively traverse the child's children
            traverse_model(child, parent=node_name, layer=layer)

    # Traverse the model
    for i, (name, module) in enumerate(model.named_children()):
        traverse_model(module, layer=name)

    # Get the supra-adjacency matrix
    matrix, nodes = pymnet.supra_adjacency_matrix(net, includeCouplings=True)

    print(matrix)
    return matrix


class SubNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=50):
        super(SubNetwork, self).__init__()

        # Define the network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x


class SineApproximator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=50, output_dim=1, num_subnets=6):
        super(SineApproximator, self).__init__()

        self.subnets = nn.ModuleList(
            [SubNetwork(input_dim, hidden_dim) for _ in range(num_subnets)]
        )

        self.fc_out = nn.Linear(num_subnets * hidden_dim, output_dim)

    def forward(self, x):
        subnet_outputs = [subnet(x) for subnet in self.subnets]
        x = torch.cat(subnet_outputs, dim=-1)
        x = self.fc_out(x)
        return x


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
        adjacency_array = model_to_pymnet_plot(model)

        plt.figure(figsize=(4, 4))
        plt.imshow(adjacency_array, cmap="gray", interpolation="none")
        plt.clim(0, 1)  # consistent scale for each frame
        plt.colorbar(label="Edge Weight")
        plt.title(f"Epoch {epoch}; Loss {loss.item():.5f}")
        plt.savefig(f"frames/frame_{epoch}.png")
        plt.close()

        images.append(imageio.imread(f"frames/frame_{epoch}.png"))


imageio.mimsave("frames/training_multinet.gif", images, loop=15, duration=50)

# Delete the individual frames
for filename in os.listdir("frames"):
    if filename.endswith(".png"):
        os.remove(f"frames/{filename}")


# # Load the models
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)

# # Generate the plots
# model_to_pymnet_plot(resnet18, "resnet18")
# # model_to_pymnet_plot(alexnet, 'alexnet')
# model_to_pymnet_plot(squeezenet, "squeezenet")
# # model_to_pymnet_plot(vgg16, 'vgg16')
# model_to_pymnet_plot(densenet, "densenet")
# model_to_pymnet_plot(inception, "inception")
# model_to_pymnet_plot(googlenet, "googlenet")
# model_to_pymnet_plot(shufflenet, "shufflenet")
# model_to_pymnet_plot(mobilenet, "mobilenet")
# model_to_pymnet_plot(resnext50_32x4d, "resnext50_32x4d")
# model_to_pymnet_plot(wide_resnet50_2, "wide_resnet50_2")
# model_to_pymnet_plot(mnasnet, "mnasnet")
