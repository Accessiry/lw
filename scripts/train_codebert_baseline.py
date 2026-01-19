from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from fcg.data import CodeDataset, load_code_samples
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


def train(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    samples = load_code_samples(dataset_root)
    dataset = CodeDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
        for encoded, labels in dataloader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            logits = classifier(**encoded)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")

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
    args = parser.parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train(args)
