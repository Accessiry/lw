import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


RELATION_NAMES = [
    "cfg",
    "dominate",
    "post_dominate",
    "ref",
]


@dataclass
class SampleIndex:
    path: str
    label: int


def _load_npz(path: str) -> Data:
    archive = np.load(path)
    x = torch.from_numpy(archive["x"]).float()
    edge_types = []
    edge_indices = []

    for relation_id, name in enumerate(RELATION_NAMES):
        key = f"edge_index_{name}"
        edge_index = torch.from_numpy(archive[key]).long()
        if edge_index.numel() == 0:
            continue
        edge_indices.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), relation_id, dtype=torch.long))

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.stats = torch.from_numpy(archive["stats"]).float()
    return data


class FunctionNPZDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.root = root
        self.split = split
        self.samples = self._index_samples()

    def _index_samples(self) -> List[SampleIndex]:
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: List[SampleIndex] = []
        for label in (0, 1):
            label_dir = os.path.join(split_dir, str(label))
            if not os.path.isdir(label_dir):
                continue
            for sample_id in os.listdir(label_dir):
                npz_path = os.path.join(label_dir, sample_id, "after.npz")
                if os.path.isfile(npz_path):
                    samples.append(SampleIndex(path=npz_path, label=label))
        if not samples:
            raise RuntimeError(f"No samples found in {split_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        sample = self.samples[index]
        data = _load_npz(sample.path)
        data.y = torch.tensor(sample.label, dtype=torch.long)
        data.sample_id = os.path.basename(os.path.dirname(sample.path))
        return data
