# Current Dataset Status

## CVC-ClinicDB

- total image/mask pairs: `612`
- generated stable split:
- train: `429`
- val: `61`
- test: `122`
- split summary file: `data/CVC-ClinicDB/splits/split_summary.json`

## Kvasir-SEG

- total image/mask pairs: `1000`
- original manifests:
- train: `880`
- val: `120`
- generated stable split:
- train: `792`
- val: `88`
- test: `120`
- split summary file: `data/Kvasir-SEG/splits/split_summary.json`

## Notes

- In the generated Kvasir setup, original `val.txt` is used as fixed `test`.
- The new Kvasir `val` split is sampled from original `train.txt` with seed `1337`.
- In the generated CVC setup, the default ratios are `70/10/20` for `train/val/test`.
