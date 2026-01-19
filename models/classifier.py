from typing import Literal, Optional

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class GraphClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_channels: int,
        encoder_out_channels: Optional[int] = None,
        num_classes: int = 2,
        pooling: Literal["mean", "max", "sum", "meanmax"] = "meanmax",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        base_channels = encoder_out_channels or hidden_channels
        pooled_dim = base_channels
        if pooling == "meanmax":
            pooled_dim = base_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self.encoder(data.x, data.edge_index, data.edge_type)
        if self.pooling == "mean":
            pooled = global_mean_pool(x, data.batch)
        elif self.pooling == "max":
            pooled = global_max_pool(x, data.batch)
        elif self.pooling == "sum":
            pooled = global_add_pool(x, data.batch)
        else:
            pooled = torch.cat(
                [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
                dim=1,
            )
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
