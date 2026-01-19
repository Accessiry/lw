import argparse
import os
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset.graph_dataset import FunctionNPZDataset
from metrics.binary import (
    BinaryMetrics,
    compute_binary_metrics,
    compute_binary_metrics_at_threshold,
)
from models.classifier import GraphClassifier
from models.rgat import RGATEncoder
from utils.logger import make_run_dir, save_json
from utils.seed import seed_worker, set_seed


def build_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    sampler: str,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = FunctionNPZDataset(root=root, split="train")
    val_dataset = FunctionNPZDataset(root=root, split="val")
    if sampler == "balanced":
        labels = torch.tensor([sample.label for sample in train_dataset.samples])
        class_counts = torch.bincount(labels, minlength=2).float()
        weights = (class_counts.sum() / (class_counts + 1e-6))[labels]
        sampler_obj = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_obj,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
    else:
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


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    if focal_gamma <= 0:
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    ce_loss = F.nll_loss(log_probs, targets, weight=class_weights, reduction="none")
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_term = (1 - pt).pow(focal_gamma)
    return (focal_term * ce_loss).mean()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    log_interval: int,
    focal_gamma: float,
    label_smoothing: float,
) -> float:
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = compute_loss(logits, batch.y, class_weights, focal_gamma, label_smoothing)
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


def find_best_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    min_threshold: float,
    max_threshold: float,
    steps: int,
) -> Tuple[float, BinaryMetrics]:
    best_threshold = 0.5
    best_metrics = compute_binary_metrics_at_threshold(probs, targets, best_threshold)
    best_score = (best_metrics.f1 + best_metrics.mcc) / 2
    for step in range(steps + 1):
        threshold = min_threshold + (max_threshold - min_threshold) * (step / steps)
        metrics = compute_binary_metrics_at_threshold(probs, targets, threshold)
        score = (metrics.f1 + metrics.mcc) / 2
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold_search: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, BinaryMetrics, float]:
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
    probs = torch.softmax(logits, dim=-1)[:, 1]
    if threshold_search:
        best_threshold, metrics = find_best_threshold(
            probs,
            targets,
            threshold_min,
            threshold_max,
            threshold_steps,
        )
    else:
        metrics = compute_binary_metrics(logits, targets)
        best_threshold = 0.5
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, metrics, best_threshold


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
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--sampler", type=str, choices=["none", "balanced"], default="balanced")
    parser.add_argument("--pooling", type=str, choices=["mean", "max", "sum", "meanmax", "attn"], default="attn")
    parser.add_argument("--edge-dropout", type=float, default=0.1)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--threshold-search", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.sampler,
    )
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
        use_layernorm=True,
        use_residual=True,
        jk_mode="cat",
        edge_dropout=args.edge_dropout,
    )
    model = GraphClassifier(
        encoder=encoder,
        hidden_channels=args.hidden_channels,
        encoder_out_channels=encoder.out_channels,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None
    class_weights = compute_class_weights(train_loader, device) if args.use_class_weights else None

    run_dir = make_run_dir(args.output_dir, args.run_name)
    best_metric = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    best_threshold = 0.5
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
            args.focal_gamma,
            args.label_smoothing,
        )
        val_loss, metrics, threshold = evaluate(
            model,
            val_loader,
            device,
            args.threshold_search,
            args.threshold_min,
            args.threshold_max,
            args.threshold_steps,
        )
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
            best_metrics = asdict(metrics)
            best_epoch = epoch
            best_threshold = threshold
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    if best_state:
        torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
        best_metrics_payload = {
            "epoch": best_epoch,
            "best_score": best_metric,
            "best_threshold": best_threshold,
            **best_metrics,
        }
        save_json(best_metrics_payload, os.path.join(run_dir, "best_metrics.json"))
        print(f"Best epoch {best_epoch} | score={best_metric:.4f}", flush=True)
        if best_metrics:
            print(
                "  [best] acc={:.4f} precision={:.4f} recall={:.4f} f1={:.4f} mcc={:.4f}".format(
                    best_metrics.get("accuracy", 0.0),
                    best_metrics.get("precision", 0.0),
                    best_metrics.get("recall", 0.0),
                    best_metrics.get("f1", 0.0),
                    best_metrics.get("mcc", 0.0),
                ),
                flush=True,
            )

    summary = {
        "best_score": best_metric,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "config": vars(args),
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))


if __name__ == "__main__":
    main()
