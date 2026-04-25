# analysis_data

This folder contains dataset utilities for the segmentation project.

## Purpose

`analysis_data/` is responsible for:

1. creating fixed `train/val/test` manifests so experiments are reproducible
2. exporting dataset statistics and summaries into `analysis_data/reports/`

## Supported Datasets

| Dataset key | Root | Archive |
|---|---|---|
| `cvc_clinicdb` | `data/CVC-ClinicDB` | `data/CVC-ClinicDB.zip` |
| `kvasir_seg` | `data/Kvasir-SEG` | `data/Kvasir-SEG.zip` |
| `etis` | `data/ETIS` | `data/ETIS.zip` |
| `cvc_colondb` | `data/CVC-ColonDB` | `data/CVC-ColonDB.zip` |
| `cvc_300` | `data/CVC-300` | `data/CVC-300.zip` |

Legacy aliases such as `cvc`, `kvasir`, `etis_larib`, `cvc_colon_db`, and `cvc300` are accepted by the utilities, but new experiments should use the canonical keys above.

## Split Policy

All supported polyp datasets use stable manifest files in `data/<dataset>/splits/`.

Default policy for CVC-ClinicDB, ETIS, CVC-ColonDB, and CVC-300:

- seed: `1337`
- train ratio: `0.8`
- validation ratio: `0.1`
- test ratio: `0.1`
- slicing rule: `train=int(total*0.8)`, `val=int(total*0.1)`, `test=remaining samples`

Kvasir-SEG policy:

- requires `data/Kvasir-SEG/train.txt` and `data/Kvasir-SEG/val.txt`
- `data/Kvasir-SEG/val.txt` is copied/promoted to `data/Kvasir-SEG/splits/test.txt`
- `data/Kvasir-SEG/train.txt` is split into train/val with a stable `90/10` split
- if either required root manifest is missing or empty, split generation stops with an error instead of scanning and splitting all images

Current expected counts:

| Dataset | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| `cvc_clinicdb` | 612 | 489 | 61 | 62 |
| `kvasir_seg` | 1000 | 792 | 88 | 120 |
| `etis` | 196 | 156 | 19 | 21 |
| `cvc_colondb` | 380 | 304 | 38 | 38 |
| `cvc_300` | 60 | 48 | 6 | 6 |

## Generate Splits

Generate or refresh all manifests:

```bash
python analysis_data/generate_splits.py --dataset all --extract --seed 1337
```

Generate one dataset:

```bash
python analysis_data/generate_splits.py --dataset cvc_clinicdb --extract --seed 1337
python analysis_data/generate_splits.py --dataset kvasir_seg --extract --seed 1337
python analysis_data/generate_splits.py --dataset etis --extract --seed 1337
python analysis_data/generate_splits.py --dataset cvc_colondb --extract --seed 1337
python analysis_data/generate_splits.py --dataset cvc_300 --extract --seed 1337
```

If all three split manifests already exist, the generator keeps them and returns `skipped: true`. Add `--overwrite` only when you intentionally want to regenerate split files.

Generated files:

```text
data/<dataset>/splits/train.txt
data/<dataset>/splits/val.txt
data/<dataset>/splits/test.txt
data/<dataset>/splits/split_summary.json
```

## Analyze Datasets

After activating the project environment with PyTorch installed:

```bash
python analysis_data/analyze_datasets.py --dataset all
```

The analyzer writes:

- `analysis_data/reports/<dataset>_summary.json`
- `analysis_data/reports/<dataset>_summary.md`
- `analysis_data/reports/dataset_overview.json`

Reports include pair counts, split counts, common image sizes, RGB mean/std, foreground ratio, mask values, and sample file paths.

The dataset `.sh` scripts do not run analysis by default. Set `PREPARE_ANALYSIS=1` if you want a run script to call `analysis_data/analyze_datasets.py` before training.

## Notes

- Training and testing scripts prefer manifests in each dataset's `splits/` folder.
- This avoids different train/val/test partitions across runs.
- If you intentionally change the seed or ratios, regenerate the split files and update this README/status report together.
