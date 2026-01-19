from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CodeSample:
    path: Path
    label: int
    text: str


def iter_c_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.c"):
        if path.is_file():
            yield path


def load_code_samples(dataset_root: Path) -> List[CodeSample]:
    samples: List[CodeSample] = []
    for label_name, label in ("novul", 0), ("vul", 1):
        folder = dataset_root / label_name
        if not folder.exists():
            continue
        for path in iter_c_files(folder):
            text = path.read_text(encoding="utf-8", errors="ignore")
            samples.append(CodeSample(path=path, label=label, text=text))
    if not samples:
        raise ValueError(f"No C files found under {dataset_root}")
    return samples


class CodeDataset(Dataset):
    def __init__(self, samples: List[CodeSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        sample = self.samples[index]
        return sample.text, torch.tensor(sample.label, dtype=torch.long)
