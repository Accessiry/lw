from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer


def cache_key(path: str) -> str:
    return sha1(path.encode("utf-8")).hexdigest()


def load_entries(jsonl_path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def embed_nodes(
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    node_texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    outputs = []
    autocast = torch.cuda.amp.autocast if use_fp16 and device.type == "cuda" else nullcontext
    with torch.no_grad():
        for batch in chunked(node_texts, batch_size):
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with autocast():
                embeddings = encoder(**tokens).last_hidden_state[:, 0, :].cpu()
            outputs.append(embeddings)
    if not outputs:
        return torch.zeros((0, encoder.config.hidden_size), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpg-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--node-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"

    entries = load_entries(Path(args.cpg_jsonl))
    if args.limit:
        entries = entries[: args.limit]

    encoder = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    with index_path.open("w", encoding="utf-8") as index_handle:
        for entry in entries:
            path = str(entry.get("path", ""))
            nodes = entry.get("nodes", [])
            node_texts = [node.get("code", "") for node in nodes]
            if not node_texts:
                node_texts = [path or "<empty>"]

            embeddings = embed_nodes(
                encoder=encoder,
                tokenizer=tokenizer,
                node_texts=node_texts,
                max_length=args.max_length,
                batch_size=args.node_batch_size,
                device=device,
                use_fp16=args.fp16,
            )
            cache_name = f"{cache_key(path)}.pt"
            torch.save({"path": path, "embeddings": embeddings}, output_dir / cache_name)
            index_handle.write(json.dumps({"path": path, "file": cache_name}) + "\n")


if __name__ == "__main__":
    main()
