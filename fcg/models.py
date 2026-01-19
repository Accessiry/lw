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


class RelationWeightedAggregator(nn.Module):
    def __init__(self, hidden_size: int, relation_count: int):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.ones(relation_count, 1))
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, graph: CPGGraph) -> torch.Tensor:
        node_embeddings = graph.node_embeddings
        src, dst = graph.edge_index
        weights = self.relation_weights[graph.edge_types].squeeze(-1)
        messages = node_embeddings[src] * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, dst, messages)
        return self.proj(aggregated)


class CPGClassifier(nn.Module):
    def __init__(self, hidden_size: int, relation_count: int, num_classes: int = 2):
        super().__init__()
        self.aggregator = RelationWeightedAggregator(hidden_size, relation_count)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, graph: CPGGraph) -> torch.Tensor:
        node_repr = self.aggregator(graph)
        graph_repr = node_repr.mean(dim=0)
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
