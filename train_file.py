import argparse
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset.graph_dataset import FunctionNPZDataset
from metrics.binary import BinaryMetrics, compute_binary_metrics
from models.classifier import GraphClassifier
from models.rgat import RGATEncoder
from utils.logger import make_run_dir, save_json
from utils.seed import seed_worker, set_seed


def build_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = FunctionNPZDataset(root=root, split="train")
    val_dataset = FunctionNPZDataset(root=root, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        if log_interval > 0 and step % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  [train] step {step}/{len(loader)} "
                f"loss={loss.item():.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return total_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, BinaryMetrics]:
    model.eval()
    losses = []
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            losses.append(loss.item() * batch.num_graphs)
            all_logits.append(logits.cpu())
            all_targets.append(batch.y.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(logits, targets)
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, metrics


def compute_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.long)
    for batch in loader:
        counts += torch.bincount(batch.y, minlength=2)
    weights = counts.float().sum() / (counts.float() + 1e-6)
    weights = weights / weights.mean()
    return weights.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Function-level vulnerability training")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="function_rgat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args.data_root, args.batch_size, args.num_workers)
    print(
        f"Loaded {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples.",
        flush=True,
    )

    encoder = RGATEncoder(
        in_channels=768,
        hidden_channels=args.hidden_channels,
        num_relations=4,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GraphClassifier(
        encoder=encoder,
        hidden_channels=args.hidden_channels,
        pooling="meanmax",
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(train_loader, device)

    run_dir = make_run_dir(args.output_dir, args.run_name)
    best_metric = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            args.log_interval,
        )
        val_loss, metrics = evaluate(model, val_loader, device)
        print(
            "  [val] loss={:.4f} acc={:.4f} precision={:.4f} recall={:.4f} "
            "f1={:.4f} mcc={:.4f}".format(
                val_loss,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1,
                metrics.mcc,
            ),
            flush=True,
        )

        metrics_payload = asdict(metrics)
        metrics_payload.update({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_json(metrics_payload, os.path.join(run_dir, f"metrics_epoch_{epoch}.json"))

        score = (metrics.f1 + metrics.mcc) / 2
        if score > best_metric:
            best_metric = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state:
        torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    summary = {
        "best_score": best_metric,
        "config": vars(args),
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))


if __name__ == "__main__":
    main()
