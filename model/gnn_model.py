import torch
import torch.nn.functional as FF
from torch_geometric.nn import SAGEConv

class fraudDetector(torch.nn.Module){
    def __intit__(self, number_of_input_feature, hidden_size= 64, number_of_classes= 2):
        super(fraudDetector, self. __init__())

        self._first_layer = SAGEConv(number_of_input_feature, hidden_size= 64) 
        self._second_layer = SAGEConv(number_of_input_feature, hidden_size= 64)

        self.output_layer= torch.nn.Linear((number_of_input_feature, hidden_size))
        self.dropout_layer= torch.nn.Dropout(p=0.3)   

    def forward(self, node_features, edge_connections):
        result = self.first_layer(node_features, edge_connections)
        result = F.relu(result)
        result = self.dropout(result)

        result = self.second_layer(result, edge_connections)
        result = F.relu(result)
        result = self.dropout(result)

        result = self.output_layer(result)
        return result
                               
}