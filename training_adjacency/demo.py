import torch
import torch.nn as nn
import torch.optim as optim

from sine_approximator import SineApproximator
from visualizer import train_and_visualize

# Instantiate the model
model = SineApproximator()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Define the 3D Rosenbrock-like function
def rosenbrock_3d(x, y, z):
    a = 1
    b = 100
    c = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2 + c * (z - y**2) ** 2


# Generate some random input data
x_values = torch.rand(1000, 1)
y_values = torch.rand(1000, 1)
z_values = torch.rand(1000, 1)

# Concatenate the x_values, y_values, and z_values to create the input
x = torch.cat([x_values, y_values, z_values], dim=-1)

# Compute the corresponding output data
y = rosenbrock_3d(x_values, y_values, z_values)

# Train and visualize
train_and_visualize(model, x, y, optimizer=optimizer, criterion=criterion)
