# Current Dataset Status

All listed datasets use stable `train/val/test` manifests under `data/<dataset>/splits/`.

Generation command:

```bash
python analysis_data/generate_splits.py --dataset all --extract --overwrite --seed 1337
```

Default split policy for CVC-ClinicDB, ETIS, CVC-ColonDB, and CVC-300:

- train: `int(total * 0.8)`
- val: `int(total * 0.1)`
- test: remaining samples

Kvasir-SEG split policy:

- requires root manifests `data/Kvasir-SEG/train.txt` and `data/Kvasir-SEG/val.txt`
- root `val.txt` is used as `splits/test.txt`
- root `train.txt` is split into train/val with a stable `90/10` split

| Dataset | Root | Total | Train | Val | Test | Summary |
|---|---|---:|---:|---:|---:|---|
| `cvc_clinicdb` | `data/CVC-ClinicDB` | 612 | 489 | 61 | 62 | `data/CVC-ClinicDB/splits/split_summary.json` |
| `kvasir_seg` | `data/Kvasir-SEG` | 1000 | 792 | 88 | 120 | `data/Kvasir-SEG/splits/split_summary.json` |
| `etis` | `data/ETIS` | 196 | 156 | 19 | 21 | `data/ETIS/splits/split_summary.json` |
| `cvc_colondb` | `data/CVC-ColonDB` | 380 | 304 | 38 | 38 | `data/CVC-ColonDB/splits/split_summary.json` |
| `cvc_300` | `data/CVC-300` | 60 | 48 | 6 | 6 | `data/CVC-300/splits/split_summary.json` |

Notes:

- `cvc` and `kvasir` are legacy aliases only. Prefer `cvc_clinicdb` and `kvasir_seg` for new outputs.
- Full statistical summaries can be regenerated with `python analysis_data/analyze_datasets.py --dataset all` after activating an environment with PyTorch installed.
