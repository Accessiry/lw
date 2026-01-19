# lw

This repo provides two starter baselines for C vulnerability classification:

1. A **CodeBERT** text classifier trained on raw C files (`vul` vs `novul`).
2. A **CPG + relation-weighted GNN** classifier that uses Joern to extract CPG JSON, encodes node contents with CodeBERT, and learns relation weights.

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

## Baseline 2: Joern + CPG + relation-weighted GNN

### 1) Extract CPG JSON

```bash
python scripts/extract_cpg_json.py \
  --joern-home /home/user/mzaj/joern-cli \
  --input-dir /home/user/mzaj/datasets/FFMpegQemu/code \
  --output-dir /home/user/mzaj/FCG/outputs/cpg
```

This produces a JSON export from Joern. Convert it into a JSONL file where each line has:

```json
{
  "nodes": [{"id": 1, "code": "..."}],
  "edges": [[0, 1, "AST"], [1, 2, "CFG"]],
  "label": 0
}
```

You can write a small converter once you decide the schema you want.

### 2) Train the CPG classifier

```bash
python scripts/train_cpg_model.py \
  --cpg-jsonl /home/user/mzaj/FCG/outputs/cpg/cpg.jsonl \
  --model-path /home/user/PycharmProjects/pythonProject/model/codebert-base \
  --output-dir /home/user/mzaj/FCG/outputs/cpg
```

## Notes

- The CPG model learns **relation weights** (AST/CFG/CDG/DDG by default) and can be extended later with deeper GNN layers or graph pooling.
- `scripts/train_cpg_model.py` assumes a JSONL graph format. Adjust `build_graph()` as needed to match your final Joern export schema.
