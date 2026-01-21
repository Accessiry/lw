# lw

This repo provides two starter baselines for C vulnerability classification:

1. A **CodeBERT** text classifier trained on raw C files (`vul` vs `novul`).
2. A **CPG + relation-weighted GNN** classifier that uses Joern to extract CPG graphson, converts it into JSONL graphs, encodes node contents with CodeBERT, and learns relation weights.

## Expected dataset layout

```
/home/user/mzaj/datasets/FFMpegQemu/code/
  vul/
    *.c
  novul/
    *.c
```

## Environment

Python 3.10 virtualenv, Linux.

## Baseline 1: CodeBERT text classifier

```bash
python scripts/train_codebert_baseline.py \
  --dataset-root /home/user/mzaj/datasets/FFMpegQemu/code \
  --model-path /home/user/PycharmProjects/pythonProject/model/codebert-base \
  --output-dir /home/user/mzaj/FCG/outputs/codebert
```

By default this splits data into train/val/test (80/10/10) with a fixed seed. Adjust with `--train-ratio`, `--val-ratio`, and `--seed`.

## Baseline 2: Joern + CPG + relation-weighted GNN

### 1) Extract CPG graphson and convert to JSONL

```bash
python scripts/extract_cpg_json.py \
  --joern-home /home/user/mzaj/joern-cli \
  --input-dir /home/user/mzaj/datasets/FFMpegQemu/code \
  --output-jsonl /home/user/mzaj/FCG/outputs/cpg/cpg.jsonl \
  --tmp-root /tmp/joern_cpg
```

The JSONL format contains one graph per C file:

```json
{
  "path": "/home/user/mzaj/datasets/FFMpegQemu/code/vul/sample.c",
  "label": 1,
  "nodes": [{"id": 1, "code": "...", "label": "CALL"}],
  "edges": [[0, 1, "AST"], [1, 2, "CFG"]]
}
```

For a quick smoke test on a few files, add `--limit 5`. To speed up extraction, increase `--workers` (each worker runs its own Joern process).
For large runs, `--workers 0` auto-uses the CPU count, `--chunksize` controls task scheduling overhead, and `--resume` appends to an existing JSONL while skipping already processed files.

### 2) Train the CPG classifier

```bash
python scripts/train_cpg_model.py \
  --cpg-jsonl /home/user/mzaj/FCG/outputs/cpg/cpg.jsonl \
  --model-path /home/user/PycharmProjects/pythonProject/model/codebert-base \
  --output-dir /home/user/mzaj/FCG/outputs/cpg
```

This also uses a default 80/10/10 train/val/test split with `--train-ratio`, `--val-ratio`, and `--seed` available.
If you hit GPU OOM, reduce `--batch-size`, lower `--max-length`, cap nodes per graph with `--max-nodes`, or enable `--fp16`.
For class imbalance or collapsed predictions, `--class-weight auto` enables inverse-frequency weights, and `--log-graph-stats` prints node/edge summary stats per split.
You can adjust the graph model capacity with `--gnn-layers`, `--dropout`, and `--pooling`, and use `--early-stop-patience` for early stopping on validation F1.
For additional stability, try `--label-smoothing`, `--focal-gamma`, and `--grad-clip`, and enable node type embeddings with `--type-embed-dim`.

### Optional: cache CodeBERT node embeddings for faster training

```bash
python scripts/cache_cpg_embeddings.py \
  --cpg-jsonl /home/user/mzaj/FCG/outputs/cpg/cpg.jsonl \
  --model-path /home/user/PycharmProjects/pythonProject/model/codebert-base \
  --output-dir /home/user/mzaj/FCG/outputs/cpg/cache \
  --max-length 128 \
  --node-batch-size 64 \
  --fp16
```

Then train using the cache:

```bash
python scripts/train_cpg_model.py \
  --cpg-jsonl /home/user/mzaj/FCG/outputs/cpg/cpg.jsonl \
  --model-path /home/user/PycharmProjects/pythonProject/model/codebert-base \
  --output-dir /home/user/mzaj/FCG/outputs/cpg \
  --feature-cache /home/user/mzaj/FCG/outputs/cpg/cache
```

## Notes

- The CPG model learns **relation weights** (AST/CFG/CDG/DDG/DOMINATE/POST_DOMINATE/REF by default) and can be extended later with deeper GNN layers or graph pooling.
- If Joern outputs additional relations, add them via `--relations` when training.
