from __future__ import annotations

import argparse
import json
import random
from hashlib import sha1
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
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


def load_cache_index(cache_dir: Path) -> Dict[str, str]:
    index_path = cache_dir / "index.jsonl"
    if not index_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "path" in obj and "file" in obj:
                mapping[str(obj["path"])] = str(obj["file"])
    return mapping


def cache_key(path: str) -> str:
    return sha1(path.encode("utf-8")).hexdigest()


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
    node_type_map: Dict[str, int],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
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
    node_type_ids = torch.tensor(
        [node_type_map.get(nodes[idx].get("label", ""), 0) for idx in keep_indices],
        dtype=torch.long,
    )

    filtered_edges = [
        edge
        for edge in edges
        if edge[2] in relation_map and edge[0] in index_map and edge[1] in index_map
    ]
    if not filtered_edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros((0,), dtype=torch.long)
        return node_texts, edge_index, edge_types, keep_indices, node_type_ids

    edge_index = torch.tensor(
        [[index_map[edge[0]] for edge in filtered_edges], [index_map[edge[1]] for edge in filtered_edges]],
        dtype=torch.long,
    )
    edge_types = torch.tensor([relation_map[edge[2]] for edge in filtered_edges], dtype=torch.long)
    return node_texts, edge_index, edge_types, keep_indices, node_type_ids


def load_cached_embeddings(entry: Dict[str, object], cache_dir: Path, cache_index: Dict[str, str]) -> torch.Tensor:
    path = str(entry.get("path", ""))
    cache_name = cache_index.get(path)
    if not cache_name:
        cache_name = f"{cache_key(path)}.pt"
    cache_path = cache_dir / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file for {path}: {cache_path}")
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    if torch.is_tensor(data):
        return data
    raise ValueError(f"Unsupported cache format at {cache_path}")


def collate_batch(
    tokenizer,
    relation_map,
    max_nodes: int,
    max_length: int,
    batch,
    cache_dir,
    cache_index,
    node_type_map,
):
    graphs = []
    labels = []
    for entry in batch:
        node_texts, edge_index, edge_types, keep_indices, node_type_ids = build_graph(
            entry,
            relation_map,
            max_nodes,
            node_type_map,
        )
        if cache_dir:
            embeddings = load_cached_embeddings(entry, cache_dir, cache_index)
            if keep_indices:
                embeddings = embeddings[keep_indices]
            graphs.append((embeddings, edge_index, edge_types, node_type_ids))
        else:
            if not node_texts:
                node_texts = [entry.get("path", "<empty>")]
            tokens = tokenizer(
                node_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            graphs.append((tokens, edge_index, edge_types, node_type_ids))
        labels.append(entry.get("label", 0))
    return graphs, torch.tensor(labels, dtype=torch.long)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.softmax(logits, dim=-1)
    preds = (probs[:, 1] >= threshold).long()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    macro_f1 = (f1 + f1_neg) / 2
    balanced_acc = (recall + recall_neg) / 2
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / denom**0.5) if denom else 0.0
    acc = correct / total if total else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "threshold": threshold,
    }


def evaluate(
    classifier: CPGClassifier,
    encoder: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    cache_dir: Path | None,
    threshold: float,
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
            for node_inputs, edge_index, edge_types, node_type_ids in graphs:
                if cache_dir:
                    node_embeddings = node_inputs.to(device)
                else:
                    tokens = {k: v.to(device) for k, v in node_inputs.items()}
                    node_embeddings = encoder(**tokens).last_hidden_state[:, 0, :]
                graph = CPGGraph(
                    node_embeddings=node_embeddings,
                    edge_index=edge_index.to(device),
                    edge_types=edge_types.to(device),
                    node_type_ids=node_type_ids.to(device),
                )
                logits_list.append(classifier(graph))
            logits = torch.stack(logits_list)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    if not all_logits:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_metrics(logits, labels, threshold=threshold)


def build_type_mapping(entries: List[Dict[str, object]]) -> Dict[str, int]:
    labels = set()
    for entry in entries:
        for node in entry.get("nodes", []):
            label = node.get("label")
            if label:
                labels.add(label)
    mapping = {"<UNK>": 0}
    for label in sorted(labels):
        mapping[label] = len(mapping)
    return mapping


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(log_probs, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    loss = (1 - pt) ** gamma * ce
    return loss.mean()


def train(args: argparse.Namespace) -> None:
    relation_map = {name: idx for idx, name in enumerate(args.relations)}
    entries = load_entries(Path(args.cpg_jsonl))
    node_type_map = build_type_mapping(entries)
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

    cache_dir = Path(args.feature_cache) if args.feature_cache else None
    cache_index = load_cache_index(cache_dir) if cache_dir else {}

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
            cache_dir,
            cache_index,
            node_type_map,
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
            cache_dir,
            cache_index,
            node_type_map,
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
            cache_dir,
            cache_index,
            node_type_map,
        ),
    )

    encoder = AutoModel.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = CPGClassifier(
        encoder.config.hidden_size,
        relation_count=len(relation_map),
        num_layers=args.gnn_layers,
        dropout=args.dropout,
        node_type_count=len(node_type_map),
        type_embed_dim=args.type_embed_dim,
        pooling=args.pooling,
    )
    classifier.to(device)
    encoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(args.early_stop_patience // 2, 1),
        min_lr=1e-6,
    )
    class_weights = None
    if args.class_weight == "auto":
        class_weights = build_class_weights(train_entries).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    classifier.train()
    best_val_f1 = -1.0
    best_metrics: dict | None = None
    best_epoch = -1
    epochs_without_improve = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "cpg_classifier_best.pt"
    best_threshold = args.threshold
    for epoch in range(args.epochs):
        total_loss = 0.0
        autocast = torch.cuda.amp.autocast if args.fp16 and device.type == "cuda" else nullcontext
        for graphs, labels in train_loader:
            labels = labels.to(device)
            logits_list = []
            for node_inputs, edge_index, edge_types, node_type_ids in graphs:
                if cache_dir:
                    node_embeddings = node_inputs.to(device)
                else:
                    tokens = {k: v.to(device) for k, v in node_inputs.items()}
                    with torch.no_grad(), autocast():
                        node_embeddings = encoder(**tokens).last_hidden_state[:, 0, :]
                graph = CPGGraph(
                    node_embeddings=node_embeddings,
                    edge_index=edge_index.to(device),
                    edge_types=edge_types.to(device),
                    node_type_ids=node_type_ids.to(device),
                )
                logits_list.append(classifier(graph))
            logits = torch.stack(logits_list)
            if args.focal_gamma > 0:
                loss = focal_loss(logits, labels, class_weights, args.focal_gamma)
            else:
                loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
        val_metrics = evaluate(
            classifier,
            encoder,
            val_loader,
            device,
            args.fp16,
            cache_dir,
            threshold=args.threshold,
        )
        if args.threshold_grid:
            best_grid_f1 = val_metrics["f1"]
            best_grid_threshold = args.threshold
            for threshold in args.threshold_grid:
                grid_metrics = evaluate(
                    classifier,
                    encoder,
                    val_loader,
                    device,
                    args.fp16,
                    cache_dir,
                    threshold=threshold,
                )
                if grid_metrics["f1"] > best_grid_f1:
                    best_grid_f1 = grid_metrics["f1"]
                    best_grid_threshold = threshold
            if best_grid_threshold != args.threshold:
                val_metrics = evaluate(
                    classifier,
                    encoder,
                    val_loader,
                    device,
                    args.fp16,
                    cache_dir,
                    threshold=best_grid_threshold,
                )
            best_threshold = best_grid_threshold
        scheduler.step(val_metrics["f1"])
        print(
            "epoch={epoch} loss={loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
            " val_precision={precision:.4f} val_recall={recall:.4f} val_macro_f1={macro_f1:.4f}"
            " val_balanced_acc={balanced_acc:.4f} val_mcc={mcc:.4f}".format(
                epoch=epoch + 1,
                loss=avg_loss,
                **val_metrics,
            )
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            torch.save(classifier.state_dict(), best_path)
            best_metrics = {
                "epoch": best_epoch,
                **val_metrics,
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if args.early_stop_patience and epochs_without_improve >= args.early_stop_patience:
                print(f"early_stop=1 best_epoch={best_epoch} best_val_f1={best_val_f1:.4f}")
                break
        classifier.train()

    if best_path.exists():
        classifier.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(
        classifier,
        encoder,
        test_loader,
        device,
        args.fp16,
        cache_dir,
        threshold=best_threshold,
    )
    print(
        "test_acc={acc:.4f} test_f1={f1:.4f} test_precision={precision:.4f} test_recall={recall:.4f}"
        " test_macro_f1={macro_f1:.4f} test_balanced_acc={balanced_acc:.4f} test_mcc={mcc:.4f}".format(
            **test_metrics
        )
    )

    classifier_path = output_dir / "cpg_classifier.pt"
    torch.save(classifier.state_dict(), classifier_path)
    print(f"saved: {classifier_path}")
    if best_metrics:
        best_metrics_path = output_dir / "cpg_best_metrics.json"
        best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        print(f"saved: {best_metrics_path}")
    final_metrics_path = output_dir / "cpg_test_metrics.json"
    final_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"saved: {final_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpg-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/cpg")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Stop if val_f1 stops improving.")
    parser.add_argument("--gnn-layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for GNN and classifier.")
    parser.add_argument("--pooling", choices=["mean", "max", "mean_max", "attention"], default="mean_max")
    parser.add_argument("--type-embed-dim", type=int, default=64, help="Node type embedding size.")
    parser.add_argument("--max-length", type=int, default=64, help="Max token length per node text.")
    parser.add_argument("--max-nodes", type=int, default=256, help="Max nodes per graph (0 disables).")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 autocast for the encoder.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE loss.")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Use focal loss when > 0.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (0 to disable).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class.")
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of thresholds to scan on validation data.",
    )
    parser.add_argument(
        "--feature-cache",
        default=None,
        help="Directory with cached node embeddings (created by cache_cpg_embeddings.py).",
    )
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
