import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_size=64, number_of_classes=2):
        super(FraudDetector, self).__init__()
        self.first_layer  = SAGEConv(in_channels, hidden_size)
        self.second_layer = SAGEConv(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, number_of_classes)
        self.dropout      = torch.nn.Dropout(p=0.3)

    def forward(self, node_features, edge_connections):
        result = self.first_layer(node_features, edge_connections)
        result = F.relu(result)
        result = self.dropout(result)
        result = self.second_layer(result, edge_connections)
        result = F.relu(result)
        result = self.dropout(result)
        result = self.output_layer(result)
        return result