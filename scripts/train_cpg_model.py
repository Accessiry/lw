from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from fcg.models import CPGClassifier, CPGGraph


class CPGJsonDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.entries: List[Dict[str, object]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.entries[index]


def build_graph(entry: Dict[str, object], relation_map: Dict[str, int]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    nodes = entry["nodes"]
    edges = entry["edges"]
    node_texts = [node.get("code", "") for node in nodes]
    edge_index = torch.tensor([[edge[0] for edge in edges], [edge[1] for edge in edges]], dtype=torch.long)
    edge_types = torch.tensor([relation_map[edge[2]] for edge in edges], dtype=torch.long)
    return node_texts, edge_index, edge_types


def collate_batch(tokenizer, relation_map, batch):
    graphs = []
    labels = []
    for entry in batch:
        node_texts, edge_index, edge_types = build_graph(entry, relation_map)
        tokens = tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        graphs.append((tokens, edge_index, edge_types))
        labels.append(entry.get("label", 0))
    return graphs, torch.tensor(labels, dtype=torch.long)


def train(args: argparse.Namespace) -> None:
    relation_map = {name: idx for idx, name in enumerate(args.relations)}
    dataset = CPGJsonDataset(Path(args.cpg_jsonl))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(args.tokenizer, relation_map, batch),
    )

    encoder = AutoModel.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = CPGClassifier(encoder.config.hidden_size, relation_count=len(relation_map))
    classifier.to(device)
    encoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for graphs, labels in dataloader:
            labels = labels.to(device)
            logits_list = []
            for tokens, edge_index, edge_types in graphs:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.no_grad():
                    node_embeddings = encoder(**tokens).last_hidden_state[:, 0, :]
                graph = CPGGraph(
                    node_embeddings=node_embeddings,
                    edge_index=edge_index.to(device),
                    edge_types=edge_types.to(device),
                )
                logits_list.append(classifier(graph))
            logits = torch.stack(logits_list)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = output_dir / "cpg_classifier.pt"
    torch.save(classifier.state_dict(), classifier_path)
    print(f"saved: {classifier_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpg-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/cpg")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--relations",
        nargs="+",
        default=["AST", "CFG", "CDG", "DDG"],
        help="Relation types to weight (must match JSONL edge type labels)",
    )
    args = parser.parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train(args)
