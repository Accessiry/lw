from dataclasses import dataclass

import torch


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> BinaryMetrics:
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
    )
