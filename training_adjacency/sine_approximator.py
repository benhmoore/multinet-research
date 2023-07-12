import torch
import torch.nn as nn


class SubNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class SineApproximator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super().__init__()
        self.subnets = nn.ModuleList(
            [SubNetwork(input_dim, hidden_dim) for _ in range(3)]
        )
        self.fc_out = nn.Linear(3 * hidden_dim, output_dim)

    def forward(self, x):
        outputs = [subnet(x) for subnet in self.subnets]
        x = torch.cat(outputs, dim=-1)
        x = self.fc_out(x)
        return x
