from typing import List

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class RGATEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.dropout(x)
        return x
