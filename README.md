# lw

Function-level vulnerability detection training utilities for the FFmpeg + Qemu Unified NPZ dataset.

## Quick start

```bash
python train_file.py \
  --data-root /home/user/mzaj/datasets/functions_npz_slow \
  --output-dir runs \
  --run-name function_rgat
```

The training script expects the dataset directory to follow the structure described in the dataset
specification, with `train/`, `val/`, and `test/` splits and `0/` and `1/` labels.

## Outputs

Each run saves metrics per epoch and the best checkpoint into a timestamped directory under `runs/`.

- `metrics_epoch_*.json`: per-epoch loss and metrics
- `best_model.pt`: best checkpoint by average of F1 and MCC
- `summary.json`: configuration and best score

## Troubleshooting

If the script appears to hang without output, try setting `--num-workers 0` and keep the default
`--log-interval` so you can see per-step progress logs. Some environments can stall on multi-process
data loading; reducing workers to zero avoids that issue.
