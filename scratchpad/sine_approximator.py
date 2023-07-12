import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class SineApproximator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(SineApproximator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        return self.fc3(x)


# Instantiate the model
model = SineApproximator()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create some sample data
# Here, assume x and y are tensors that contain your 3D input data and target output data respectively.
x = torch.rand(100, 3) * 6.28 - 3.14  # Sample uniformly within [-3.14, 3.14]
y = torch.sin(x).sum(dim=1, keepdim=True)  # Sum of sines as target function

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()  # zero the gradient buffers
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()  # Does the update

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
