import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import matplotlib.pyplot as plt


# Define a simple MLP
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 16)
        self.fc4 = torch.nn.Linear(16, 16)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def create_graph(module, parent=None):
    G = nx.DiGraph()
    traverse_model(module, G, parent)
    adjacency_matrix = nx.adjacency_matrix(G)
    return adjacency_matrix.toarray()


def traverse_model(module, G, parent=None):
    for name, child in module.named_children():
        node_name = f"{name}_{id(child)}"
        G.add_node(node_name)

        if parent is not None:
            edge_weight = 0
            if len(list(child.parameters())) > 0:
                params = torch.cat([x.view(-1) for x in child.parameters()])
                edge_weight = torch.mean(torch.abs(params)).item()

            if G.has_edge(parent, node_name):
                G[parent][node_name]["weight"] = edge_weight
            else:
                G.add_edge(parent, node_name, weight=edge_weight)

        traverse_model(child, G, parent=node_name)


# Load the untrained model
model = MLP()

# Small synthetic dataset for training
X_train = torch.randn(100, 16)  # 100 samples, 16 features each
y_train = torch.randn(100, 16)  # 100 random target values
trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=10)

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(5):  # 5 epochs
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Every 10 iterations, save the adjacency matrix
        if i % 10 == 0:
            adjacency_array = create_graph(model)

            plt.figure(figsize=(6, 6))
            plt.imshow(adjacency_array, cmap="Greys", interpolation="none")
            plt.colorbar(label="Edge Weight")
            plt.title(f"Adjacency Matrix at epoch {epoch} iteration {i}")
            plt.savefig(f"adjacency_epoch{epoch}_iter{i}.png")
            plt.close()

print("Finished Training")
