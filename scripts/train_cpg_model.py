from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
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


def summarize_graphs(entries: List[Dict[str, object]]) -> Dict[str, float]:
    node_counts = [len(entry.get("nodes", [])) for entry in entries]
    edge_counts = [len(entry.get("edges", [])) for entry in entries]
    empty_nodes = sum(1 for count in node_counts if count == 0)
    empty_edges = sum(1 for count in edge_counts if count == 0)
    return {
        "graphs": float(len(entries)),
        "nodes_min": float(min(node_counts)) if node_counts else 0.0,
        "nodes_max": float(max(node_counts)) if node_counts else 0.0,
        "nodes_mean": float(mean(node_counts)) if node_counts else 0.0,
        "edges_min": float(min(edge_counts)) if edge_counts else 0.0,
        "edges_max": float(max(edge_counts)) if edge_counts else 0.0,
        "edges_mean": float(mean(edge_counts)) if edge_counts else 0.0,
        "empty_nodes": float(empty_nodes),
        "empty_edges": float(empty_edges),
    }


def build_class_weights(entries: List[Dict[str, object]]) -> torch.Tensor:
    counts = {0: 0, 1: 0}
    for entry in entries:
        label = int(entry.get("label", 0))
        if label in counts:
            counts[label] += 1
    total = counts[0] + counts[1]
    if total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    weight_0 = total / max(counts[0], 1)
    weight_1 = total / max(counts[1], 1)
    return torch.tensor([weight_0, weight_1], dtype=torch.float32)


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
    max_nodes: int,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    nodes = entry["nodes"]
    edges = entry["edges"]
    total_nodes = len(nodes)
    keep_indices = list(range(total_nodes))

    if max_nodes > 0 and total_nodes > max_nodes:
        seed = entry.get("path", "")
        rng = random.Random(seed)
        keep_indices = sorted(rng.sample(keep_indices, max_nodes))

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    node_texts = [nodes[idx].get("code", "") for idx in keep_indices]

    filtered_edges = [
        edge
        for edge in edges
        if edge[2] in relation_map and edge[0] in index_map and edge[1] in index_map
    ]
    if not filtered_edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros((0,), dtype=torch.long)
        return node_texts, edge_index, edge_types

    edge_index = torch.tensor(
        [[index_map[edge[0]] for edge in filtered_edges], [index_map[edge[1]] for edge in filtered_edges]],
        dtype=torch.long,
    )
    edge_types = torch.tensor([relation_map[edge[2]] for edge in filtered_edges], dtype=torch.long)
    return node_texts, edge_index, edge_types


def collate_batch(tokenizer, relation_map, max_nodes: int, max_length: int, batch):
    graphs = []
    labels = []
    for entry in batch:
        node_texts, edge_index, edge_types = build_graph(entry, relation_map, max_nodes)
        if not node_texts:
            node_texts = [entry.get("path", "<empty>")]
        tokens = tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
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
    use_fp16: bool,
) -> dict:
    classifier.eval()
    encoder.eval()
    all_logits = []
    all_labels = []
    autocast = torch.cuda.amp.autocast if use_fp16 and device.type == "cuda" else nullcontext
    with torch.no_grad(), autocast():
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
    if args.log_graph_stats:
        for name, subset in (("train", train_entries), ("val", val_entries), ("test", test_entries)):
            stats = summarize_graphs(subset)
            print(
                f"[{name}] graphs={int(stats['graphs'])} nodes(min/mean/max)={stats['nodes_min']:.0f}"
                f"/{stats['nodes_mean']:.1f}/{stats['nodes_max']:.0f} edges(min/mean/max)={stats['edges_min']:.0f}"
                f"/{stats['edges_mean']:.1f}/{stats['edges_max']:.0f} empty_nodes={int(stats['empty_nodes'])}"
                f" empty_edges={int(stats['empty_edges'])}"
            )

    train_loader = DataLoader(
        CPGJsonDataset(train_entries),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(
            args.tokenizer,
            relation_map,
            args.max_nodes,
            args.max_length,
            batch,
        ),
    )
    val_loader = DataLoader(
        CPGJsonDataset(val_entries),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            args.tokenizer,
            relation_map,
            args.max_nodes,
            args.max_length,
            batch,
        ),
    )
    test_loader = DataLoader(
        CPGJsonDataset(test_entries),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            args.tokenizer,
            relation_map,
            args.max_nodes,
            args.max_length,
            batch,
        ),
    )

    encoder = AutoModel.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = CPGClassifier(encoder.config.hidden_size, relation_count=len(relation_map))
    classifier.to(device)
    encoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    class_weights = None
    if args.class_weight == "auto":
        class_weights = build_class_weights(train_entries).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    classifier.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        autocast = torch.cuda.amp.autocast if args.fp16 and device.type == "cuda" else nullcontext
        for graphs, labels in train_loader:
            labels = labels.to(device)
            logits_list = []
            for tokens, edge_index, edge_types in graphs:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.no_grad(), autocast():
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
        val_metrics = evaluate(classifier, encoder, val_loader, device, args.fp16)
        print(
            "epoch={epoch} loss={loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
            " val_precision={precision:.4f} val_recall={recall:.4f}".format(
                epoch=epoch + 1,
                loss=avg_loss,
                **val_metrics,
            )
        )
        classifier.train()

    test_metrics = evaluate(classifier, encoder, test_loader, device, args.fp16)
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
    parser.add_argument("--max-length", type=int, default=64, help="Max token length per node text.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Max nodes per graph (0 disables).")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 autocast for the encoder.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "auto"],
        default="auto",
        help="Class weighting strategy for imbalanced data.",
    )
    parser.add_argument(
        "--log-graph-stats",
        action="store_true",
        help="Print per-split node/edge statistics before training.",
    )
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
