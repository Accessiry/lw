from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from fcg.data import CodeDataset, CodeSample, load_code_samples
from fcg.models import CodeBERTClassifier


def collate_batch(tokenizer, batch):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return encoded, torch.stack(list(labels))


def split_samples(
    samples: List[CodeSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[CodeSample], List[CodeSample], List[CodeSample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


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


def evaluate(model: CodeBERTClassifier, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for encoded, labels in dataloader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            logits = model(**encoded)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    if not all_logits:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_metrics(logits, labels)


def train(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    samples = load_code_samples(dataset_root)
    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        CodeDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(args.tokenizer, batch),
    )
    val_loader = DataLoader(
        CodeDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(args.tokenizer, batch),
    )
    test_loader = DataLoader(
        CodeDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(args.tokenizer, batch),
    )

    encoder = AutoModel.from_pretrained(args.model_path)
    classifier = CodeBERTClassifier(encoder, hidden_size=encoder.config.hidden_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for encoded, labels in train_loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            logits = classifier(**encoded)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
        val_metrics = evaluate(classifier, val_loader, device)
        print(
            "epoch={epoch} loss={loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}"
            " val_precision={precision:.4f} val_recall={recall:.4f}".format(
                epoch=epoch + 1,
                loss=avg_loss,
                **val_metrics,
            )
        )
        classifier.train()

    test_metrics = evaluate(classifier, test_loader, device)
    print(
        "test_acc={acc:.4f} test_f1={f1:.4f} test_precision={precision:.4f} test_recall={recall:.4f}".format(
            **test_metrics
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = output_dir / "codebert_classifier.pt"
    torch.save(classifier.state_dict(), classifier_path)
    print(f"saved: {classifier_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/codebert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train(args)
