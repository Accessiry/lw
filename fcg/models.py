from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class CPGGraph:
    node_embeddings: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor


class RelationWeightedLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        relation_count: int,
        dropout: float,
        use_residual: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.ones(relation_count, 1))
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.add_self_loops = add_self_loops

    def forward(self, graph: CPGGraph, node_embeddings: torch.Tensor) -> torch.Tensor:
        src, dst = graph.edge_index
        weights = self.relation_weights[graph.edge_types].squeeze(-1)
        messages = node_embeddings[src] * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, dst, messages)

        if self.add_self_loops:
            aggregated = aggregated + node_embeddings

        if aggregated.numel() > 0:
            degree = torch.zeros(aggregated.shape[0], device=aggregated.device)
            ones = torch.ones_like(dst, dtype=degree.dtype)
            degree.index_add_(0, dst, ones)
            if self.add_self_loops:
                degree = degree + 1.0
            aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(-1)

        out = self.proj(aggregated)
        out = self.norm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + node_embeddings
        return out


class RelationWeightedEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        relation_count: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationWeightedLayer(
                    hidden_size=hidden_size,
                    relation_count=relation_count,
                    dropout=dropout,
                )
                for _ in range(max(num_layers, 1))
            ]
        )

    def forward(self, graph: CPGGraph) -> torch.Tensor:
        node_embeddings = graph.node_embeddings
        for layer in self.layers:
            node_embeddings = layer(graph, node_embeddings)
        return node_embeddings


class CPGClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        relation_count: int,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = RelationWeightedEncoder(
            hidden_size=hidden_size,
            relation_count=relation_count,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, graph: CPGGraph) -> torch.Tensor:
        node_repr = self.encoder(graph)
        mean_pool = node_repr.mean(dim=0)
        max_pool = node_repr.max(dim=0).values
        graph_repr = self.dropout(torch.cat([mean_pool, max_pool], dim=0))
        return self.classifier(graph_repr)


class CodeBERTClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def build_relation_mapping(relations: Tuple[str, ...]) -> Dict[str, int]:
    return {rel: idx for idx, rel in enumerate(relations)}
