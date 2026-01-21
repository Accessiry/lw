from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class CPGGraph:
    node_embeddings: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    node_type_ids: Optional[torch.Tensor] = None


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
        node_type_count: int = 0,
        type_embed_dim: int = 0,
        pooling: str = "mean_max",
    ):
        super().__init__()
        self.pooling = pooling
        self.type_embed_dim = type_embed_dim if node_type_count > 0 else 0
        if self.type_embed_dim > 0:
            self.type_embedding = nn.Embedding(node_type_count, self.type_embed_dim)
            self.input_proj = nn.Linear(hidden_size + self.type_embed_dim, hidden_size)
        else:
            self.type_embedding = None
            self.input_proj = None
        self.encoder = RelationWeightedEncoder(
            hidden_size=hidden_size,
            relation_count=relation_count,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        if self.pooling == "attention":
            self.attn = nn.Linear(hidden_size, 1)
            pooled_dim = hidden_size
        elif self.pooling == "mean":
            self.attn = None
            pooled_dim = hidden_size
        elif self.pooling == "max":
            self.attn = None
            pooled_dim = hidden_size
        else:
            self.attn = None
            pooled_dim = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, graph: CPGGraph) -> torch.Tensor:
        node_repr = graph.node_embeddings
        if self.type_embedding is not None and graph.node_type_ids is not None:
            type_embeds = self.type_embedding(graph.node_type_ids)
            node_repr = torch.cat([node_repr, type_embeds], dim=-1)
            node_repr = self.input_proj(node_repr)
        graph = CPGGraph(
            node_embeddings=node_repr,
            edge_index=graph.edge_index,
            edge_types=graph.edge_types,
        )
        node_repr = self.encoder(graph)
        if self.pooling == "attention":
            attn_scores = self.attn(node_repr).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=0)
            graph_repr = (attn_weights.unsqueeze(-1) * node_repr).sum(dim=0)
        elif self.pooling == "mean":
            graph_repr = node_repr.mean(dim=0)
        elif self.pooling == "max":
            graph_repr = node_repr.max(dim=0).values
        else:
            mean_pool = node_repr.mean(dim=0)
            max_pool = node_repr.max(dim=0).values
            graph_repr = torch.cat([mean_pool, max_pool], dim=0)
        graph_repr = self.dropout(graph_repr)
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
