import torch
from torch import nn
from torch_geometric.nn import JumpingKnowledge, RGCNConv
from torch_geometric.utils import dropout_edge


class RGATEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_residual: bool = True,
        jk_mode: str = "cat",
        edge_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.edge_dropout = edge_dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity())
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)
        self.jk_mode = jk_mode
        self.out_channels = hidden_channels * num_layers if jk_mode == "cat" else hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        outputs = []
        for conv in self.convs:
            if self.edge_dropout > 0:
                edge_index_dropped, edge_mask = dropout_edge(edge_index, p=self.edge_dropout)
                edge_type_dropped = edge_type[edge_mask]
                out = conv(x, edge_index_dropped, edge_type_dropped)
            else:
                out = conv(x, edge_index, edge_type)
            out = torch.relu(out)
            out = self.dropout(out)
            out = self.norms[len(outputs)](out)
            if self.use_residual and out.shape == x.shape:
                out = out + x
            x = out
            outputs.append(x)
        if self.jk_mode in {"cat", "max", "lstm"}:
            return self.jk(outputs)
        return x
