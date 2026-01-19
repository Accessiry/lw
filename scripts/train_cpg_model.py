from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from fcg.models import CPGClassifier, CPGGraph


class CPGJsonDataset(Dataset):
    def __init__(self, entries: List[Dict[str, object]]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.entries[index]


def load_entries(jsonl_path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entries.append(json.loads(line))
    return entries


def split_entries(
    entries: List[Dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def build_graph(
    entry: Dict[str, object],
    relation_map: Dict[str, int],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    nodes = entry["nodes"]
    edges = entry["edges"]
    node_texts = [node.get("code", "") for node in nodes]

    filtered_edges = [edge for edge in edges if edge[2] in relation_map]
    if not filtered_edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros((0,), dtype=torch.long)
        return node_texts, edge_index, edge_types

    edge_index = torch.tensor(
        [[edge[0] for edge in filtered_edges], [edge[1] for edge in filtered_edges]],
        dtype=torch.long,
    )
    edge_types = torch.tensor([relation_map[edge[2]] for edge in filtered_edges], dtype=torch.long)
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


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = correct / total if total else 0.0
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    classifier: CPGClassifier,
    encoder: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    classifier.eval()
    encoder.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for graphs, labels in dataloader:
            labels = labels.to(device)
            logits_list = []
            for tokens, edge_index, edge_types in graphs:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                node_embeddings = encoder(**tokens).last_hidden_state[:, 0, :]
                graph = CPGGraph(
                    node_embeddings=node_embeddings,
                    edge_index=edge_index.to(device),
                    edge_types=edge_types.to(device),
                )
                logits_list.append(classifier(graph))
            logits = torch.stack(logits_list)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    if not all_logits:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_metrics(logits, labels)


def train(args: argparse.Namespace) -> None:
    relation_map = {name: idx for idx, name in enumerate(args.relations)}
    entries = load_entries(Path(args.cpg_jsonl))
    train_entries, val_entries, test_entries = split_entries(
        entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        CPGJsonDataset(train_entries),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(args.tokenizer, relation_map, batch),
    )
    val_loader = DataLoader(
        CPGJsonDataset(val_entries),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(args.tokenizer, relation_map, batch),
    )
    test_loader = DataLoader(
        CPGJsonDataset(test_entries),
        batch_size=args.batch_size,
        shuffle=False,
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
        for graphs, labels in train_loader:
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
        avg_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
        val_metrics = evaluate(classifier, encoder, val_loader, device)
        print(
            "epoch={epoch} loss={loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
            " val_precision={precision:.4f} val_recall={recall:.4f}".format(
                epoch=epoch + 1,
                loss=avg_loss,
                **val_metrics,
            )
        )
        classifier.train()

    test_metrics = evaluate(classifier, encoder, test_loader, device)
    print(
        "test_acc={acc:.4f} test_f1={f1:.4f} test_precision={precision:.4f} test_recall={recall:.4f}".format(
            **test_metrics
        )
    )

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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--relations",
        nargs="+",
        default=["AST", "CFG", "CDG", "DDG", "DOMINATE", "POST_DOMINATE", "REF"],
        help="Relation types to weight (must match JSONL edge type labels)",
    )
    args = parser.parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train(args)
