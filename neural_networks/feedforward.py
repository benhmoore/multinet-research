import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import graphviz
from torchviz import make_dot

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

# Example dataset for training XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert data to PyTorch tensors
X = torch.Tensor(X)
y = torch.Tensor(y)

# Create an instance of the NeuralNetwork class
model = NeuralNetwork(input_size=2, hidden_size=1, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Generate a dot graph of the model
inputs = torch.randn(1, 2)
dot_graph = make_dot(model(inputs), params=dict(model.named_parameters()))

# Remove unwanted nodes from the dot representation
dot = dot_graph.source
dot = dot.replace('Param', '').replace('fcl.', '')

# Create a graph from the modified dot representation
graph = graphviz.Source(dot)

# Save the graph as an image file
graph.format = 'png'
graph.render("neural_network", view=True, cleanup=True, directory=".")

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    output = model(X)
    
    # Compute loss
    loss = criterion(output, y)

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss
    if epoch % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Make predictions
with torch.no_grad():
    predictions = model(X)
    print("Predictions:", predictions)
