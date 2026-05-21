# Medical Image Segmentation Codebase

## Current PGD-UNet update notes

The main proposal pipeline is now focused on structural pruning plus optional
teacher output distillation. The default PGD teacher is `unet_plus_plus`
(`segmentation_models_pytorch.UnetPlusPlus`) with a ResNet152 encoder. Learnable
gates are kept only as backward-compatible no-op methods; the default student
path does not use gate sparsity.

Main student training modes:

- Segmentation only: `--use_kd_output 0`
- Segmentation + KD: `--use_kd_output 1 --lambda_distill <value>`

The supported pruning strategies are:

| Strategy | Method | Display |
|---|---|---|
| `S1` | `static` | Static Blueprint |
| `S2` | `kneedle` | Kneedle Blueprint |
| `S3` | `otsu` | Otsu Blueprint |
| `S4` | `gmm` | GMM Blueprint |
| `S5` | `middle_static` | Middle-Static Conv2 |
| `S6` | `middle_kneedle` | Middle-Kneedle Conv2 |
| `S7` | `middle_otsu` | Middle-Otsu Conv2 |
| `S8` | `middle_gmm` | Middle-GMM Conv2 |
| `S9` | `full_static` | Full-Static Block |
| `S10` | `full_kneedle` | Full-Kneedle Block |
| `S11` | `full_otsu` | Full-Otsu Block |
| `S12` | `full_gmm` | Full-GMM Block |

S5-S8 keep ResNet bottleneck boundaries full and prune only `conv2` inside the
UNet++ ResNet152 encoder. S9-S12 prune the full bottleneck main path, including
`conv3` output. For UNet++, the encoder is pruned and stage projection layers
restore decoder feature widths so the UNet++ decoder remains shape-compatible.

Visualization panels now include explicit `Image`, `GT`, and `PR` labels.
Student `checkpoints/train_log.csv` includes stable validation columns for
plotting, including `val_macro_dice`, `val_iou`, `val_hd95`,
`learning_rate`, `best_val_dice`, and `is_best`.

Codebase PyTorch cho bÃ i toÃ¡n medical image segmentation 2D, Ä‘Æ°á»£c tá»• chá»©c theo 2 nhÃ¡nh rÃµ rÃ ng:

- `basic branch`: train vÃ  benchmark cÃ¡c baseline nhÆ° `unet`, `unet_resnet152`, `resunet`, `vnet`, `unetr`
- `proposal branch`: pipeline PGD gá»“m `teacher -> pruning -> student`, vá»›i default teacher `unet_plus_plus` vÃ  output chuáº©n náº±m dÆ°á»›i `outputs/pgd_unet/...`

Má»¥c tiÃªu cá»§a repo lÃ  chuáº©n hÃ³a:

- interface output cá»§a model
- output directory
- checkpoint / weight management
- metrics / reports / artifacts
- so sÃ¡nh cÃ´ng báº±ng giá»¯a baseline vÃ  proposal

## 1. Cáº¥u trÃºc chÃ­nh

```text
Code_main/
â”œâ”€ dataloaders/
â”‚  â””â”€ dataset.py
â”œâ”€ networks/
â”‚  â”œâ”€ Basic_Model/
â”‚  â”œâ”€ PGD_Unet/
â”‚  â”‚  â”œâ”€ blueprint_unet_plus_plus.py
â”‚  â”‚  â”œâ”€ gated_unet.py
â”‚  â”‚  â”œâ”€ middle_pruned_unet_plus_plus.py
â”‚  â”‚  â”œâ”€ full_pruning_unet_plus_plus.py
â”‚  â”‚  â”œâ”€ pruning.py
â”‚  â”‚  â”œâ”€ pruning_algorithms/
â”‚  â”‚  â”‚  â””â”€ Kneedle_Otsu_GMM.py
â”‚  â”‚  â””â”€ prunning.py
â”‚  â””â”€ net_factory.py
â”œâ”€ utils/
â”‚  â”œâ”€ channel_analysis.py
â”‚  â”œâ”€ checkpoint_resolver.py
â”‚  â”œâ”€ checkpoints.py
â”‚  â”œâ”€ compression_loss.py
â”‚  â”œâ”€ evaluation.py
â”‚  â”œâ”€ experiment.py
â”‚  â”œâ”€ losses.py
â”‚  â”œâ”€ model_output.py
â”‚  â”œâ”€ profiling.py
â”‚  â”œâ”€ reporting.py
â”‚  â”œâ”€ val_2d.py
â”‚  â””â”€ visualization.py
â”œâ”€ train_basic_model.py
â”œâ”€ train_pgd.py
â”œâ”€ test2d.py
â”œâ”€ compare_artifacts.py
â””â”€ outputs/
```

LÆ°u Ã½:

- `networks/net_factory.py` chá»‰ route cho `basic branch`
- `networks/PGD_Unet/prunning.py` Ä‘Æ°á»£c giá»¯ láº¡i Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch import cÅ©

## 2. Chuáº©n output model

Má»i model Ä‘á»u Ä‘Æ°á»£c chuáº©n hÃ³a qua `SegmentationModelOutput` trong `utils/model_output.py`.

Output chuáº©n gá»“m:

- `logits`
- `probs`
- `preds`
- `features`
- `aux`
- `model_name`
- `backbone_name`
- `student_name`
- `phase_name`

Äiá»u nÃ y giÃºp `train_basic_model.py`, `train_pgd.py`, `test2d.py`, pháº§n evaluate vÃ  pháº§n export artifact dÃ¹ng láº¡i Ä‘Æ°á»£c cÃ¹ng logic.

## 3. MÃ´i trÆ°á»ng

Khuyáº¿n nghá»‹:

- `Python 3.10+`
- `PyTorch` theo mÃ´i trÆ°á»ng GPU báº¡n Ä‘ang dÃ¹ng

CÃ i dependency:

```bash
pip install -r requirements.txt
```

Náº¿u muá»‘n tÃ­nh FLOPs:

```bash
pip install thop
```

## 4. Dataset

Repo hiá»‡n há»— trá»£ cÃ¡c dataset key:

- `cvc_clinicdb`
- `kvasir_seg`
- `etis`
- `etis_larib`
- `cvc_colondb`
- `cvc_colon_db`
- `cvc_300`
- `cvc300`
- `cvc` vÃ  `kvasir` váº«n cÃ²n lÃ  alias tÆ°Æ¡ng thÃ­ch cÅ©, nhÆ°ng output má»›i nÃªn dÃ¹ng `cvc_clinicdb` vÃ  `kvasir_seg`
- `generic`

VÃ­ dá»¥ cáº¥u trÃºc:

```text
data/
â””â”€ Kvasir-SEG/
   â”œâ”€ images/
   â”œâ”€ masks/
   â”œâ”€ train.txt
   â”œâ”€ val.txt
   â””â”€ test.txt
```

`dataloaders/dataset.py` há»— trá»£:

- quÃ©t cáº·p `image/mask`
- split file `train.txt`, `val.txt`, `test.txt`
- normalize mask nhá»‹ phÃ¢n vá» `0/1`
- transform train / eval riÃªng

### Stable split manifests

Chuáº©n hÃ³a split cá»‘ Ä‘á»‹nh:

```bash
python analysis_data/generate_splits.py --dataset all --extract --seed 1337
```

Script nÃ y táº¡o manifest trong `data/<dataset>/splits/`. Máº·c Ä‘á»‹nh vá»›i CVC-ClinicDB, ETIS, CVC-ColonDB vÃ  CVC-300: `train=int(total*0.8)`, `val=int(total*0.1)`, `test` láº¥y pháº§n cÃ²n láº¡i. CÃ¡c dataset polyp hiá»‡n dÃ¹ng chung split manifest Ä‘á»ƒ trÃ¡nh má»—i láº§n cháº¡y tá»± chia khÃ¡c nhau.

RiÃªng `kvasir_seg` dÃ¹ng policy theo manifest gá»‘c: báº¯t buá»™c cÃ³ `data/Kvasir-SEG/train.txt` vÃ  `data/Kvasir-SEG/val.txt`; `val.txt` gá»‘c Ä‘Æ°á»£c dÃ¹ng lÃ m `splits/test.txt`, cÃ²n `train.txt` gá»‘c Ä‘Æ°á»£c chia á»•n Ä‘á»‹nh `90/10` thÃ nh `splits/train.txt` vÃ  `splits/val.txt`. Náº¿u thiáº¿u file manifest gá»‘c, script sáº½ dá»«ng lá»—i thay vÃ¬ tá»± quÃ©t vÃ  chia toÃ n bá»™ áº£nh.

Náº¿u split Ä‘Ã£ tá»“n táº¡i, script sáº½ giá»¯ nguyÃªn vÃ  bÃ¡o skip. Chá»‰ thÃªm `--overwrite` khi báº¡n cá»‘ Ã½ muá»‘n táº¡o láº¡i split.

## 5. Basic branch

Model basic hien duoc dang ky qua `networks/net_factory.py`:

- `unet`: U-Net 2D co ban
- `resunet`: Residual U-Net
- `vnet`: V-Net 2D
- `unetr`: UNETR 2D
   - `unet_resnet152`: U-Net decoder voi ResNet152 encoder
- `att_unet`: Attention U-Net, dua theo reference trong `networks/Basic_Model/src_ref/Image_Segmentation`
- `r2unet`: R2U-Net/Recurrent Residual U-Net, dua theo reference trong `networks/Basic_Model/src_ref/Image_Segmentation`
- `unet_plus_plus`: UNet++ wrapper tu `segmentation_models_pytorch`

Voi `unet_resnet152` va `unet_plus_plus`, `--encoder_pretrained 1` se dung pretrained encoder neu dependency/weights san sang; dung `--encoder_pretrained 0` de tat.

### Train

VÃ­ dá»¥:

```bash
python train_basic_model.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

VÃ­ dá»¥ vá»›i `unet_resnet152`:

```bash
python train_basic_model.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

LÆ°u Ã½:

- vá»›i `unet_resnet152`, `encoder_pretrained` hiá»‡n máº·c Ä‘á»‹nh lÃ  `1`
- nghÄ©a lÃ  náº¿u báº¡n khÃ´ng truyá»n cá» nÃ y, repo váº«n sáº½ build encoder ResNet152 tá»« pretrained weights

### Evaluate

```bash
python test2d.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --split test
```

### Checkpoint behavior

`train_basic_model.py` sáº½:

- kiá»ƒm tra checkpoint compatible trong `outputs/<model>/<dataset>/checkpoints/`
- náº¿u cÃ³ vÃ  khÃ´ng báº­t `--force_retrain 1` thÃ¬ load láº¡i, skip train, rá»“i váº«n export final evaluation / reports / artifacts
- náº¿u chÆ°a cÃ³ checkpoint phÃ¹ há»£p thÃ¬ má»›i train

## 6. Proposal branch: PDG-UNet

### Train full pipeline

```bash
python run_pgd.py --dataset kvasir_seg --root_path data/Kvasir-SEG --teacher_model unet_plus_plus --exp pdg_kvasir_seg --max_epochs_teacher 50 --max_epochs_student 100 --prune_strategy S1 --prune_method static --static_prune_ratio 0.5 --use_kd_output 1 --lambda_distill 0.3 --batch_size 8 --patch_size 256 256
```

LÆ°u Ã½:

- vá»›i `TEACHER_MODEL=unet_plus_plus`, `--encoder_pretrained` hiá»‡n máº·c Ä‘á»‹nh lÃ  `1`
- nÃªn náº¿u teacher pháº£i train láº¡i tá»« Ä‘áº§u á»Ÿ step 1, encoder ResNet152 sáº½ máº·c Ä‘á»‹nh dÃ¹ng pretrained weights
- náº¿u báº¡n muá»‘n táº¯t pretrained vÃ  train teacher tá»« random backbone, hÃ£y truyá»n `--encoder_pretrained 0`

### Pruning strategy S1-S12

Pipeline hiá»‡n há»— trá»£ 12 strategy á»Ÿ step 2:

| Strategy | Method ná»™i bá»™ | Ã nghÄ©a | Cáº§n prune rate |
|---|---|---|---|
| `S1` | `static` | giá»¯ top-k channel theo tá»‰ lá»‡ cá»‘ Ä‘á»‹nh | cÃ³, qua `PRUNE_RATE` hoáº·c `--static_prune_ratio` |
| `S2` | `kneedle` | dynamic threshold báº±ng Kneedle | khÃ´ng |
| `S3` | `otsu` | dynamic threshold báº±ng Otsu | khÃ´ng |
| `S4` | `gmm` | dynamic threshold báº±ng Gaussian Mixture Model | khÃ´ng |
| `S5` | `middle_static` | trong ResNet152 bottleneck, giá»¯ boundary layer full vÃ  chá»‰ prune `conv2` báº±ng static top-k nhÆ° S1 | cÃ³, qua `PRUNE_RATE` hoáº·c `--static_prune_ratio` |
| `S6` | `middle_kneedle` | giá»‘ng S5 vá» cáº¥u trÃºc block, nhÆ°ng mask cá»§a `conv2` láº¥y báº±ng Kneedle riÃªng cho tá»«ng bottleneck nhÆ° S2 | khÃ´ng |
| `S7` | `middle_otsu` | giá»‘ng S5 vá» cáº¥u trÃºc block, nhÆ°ng mask cá»§a `conv2` láº¥y báº±ng Otsu riÃªng cho tá»«ng bottleneck nhÆ° S3 | khÃ´ng |
| `S8` | `middle_gmm` | giá»‘ng S5 vá» cáº¥u trÃºc block, nhÆ°ng mask cá»§a `conv2` láº¥y báº±ng GMM riÃªng cho tá»«ng bottleneck nhÆ° S4 | khÃ´ng |
| `S9` | `full_static` | prune full bottleneck path, gá»“m cáº£ `conv3 output`, báº±ng static top-k; rebuild residual/decoder shape | cÃ³, qua `PRUNE_RATE` hoáº·c `--static_prune_ratio` |
| `S10` | `full_kneedle` | giá»‘ng S9 nhÆ°ng chá»n channels báº±ng Kneedle | khÃ´ng |
| `S11` | `full_otsu` | giá»‘ng S9 nhÆ°ng chá»n channels báº±ng Otsu | khÃ´ng |
| `S12` | `full_gmm` | giá»‘ng S9 nhÆ°ng chá»n channels báº±ng GMM | khÃ´ng |

Vá»›i `S1/static`, `S5/middle_static`, hoáº·c `S9/full_static`, náº¿u prune rate lÃ  `r`, má»—i layer Ä‘Æ°á»£c prune sáº½ giá»¯:

```text
ceil((1 - r) * num_channels)
```

channel cÃ³ importance score cao nháº¥t. `static_prune_ratio` pháº£i náº±m trong `[0, 1)` vÃ  luÃ´n giá»¯ Ã­t nháº¥t 1 channel.

RiÃªng `S5/middle_static`, `S6/middle_kneedle`, `S7/middle_otsu` vÃ  `S8/middle_gmm` hiá»‡n Ä‘Æ°á»£c thiáº¿t káº¿ cho `TEACHER_MODEL=unet_plus_plus`. Vá»›i má»—i bottleneck block trong `model.encoder.layer1/layer2/layer3/layer4`, pipeline dÃ¹ng `conv2` lÃ m layer giá»¯a Ä‘á»ƒ prune. `conv1`/`bn1`, output cá»§a `conv3`/`bn3`, downsample vÃ  shape residual Ä‘Æ°á»£c giá»¯ full Ä‘á»ƒ Ä‘áº§u/cuá»‘i block khÃ´ng bá»‹ cáº¯t. VÃ¬ `conv2` output bá»‹ prune tháº­t, `conv3` sáº½ chá»‰ copy input slice tÆ°Æ¡ng á»©ng cÃ¡c channel `conv2` Ä‘Æ°á»£c giá»¯, nhÆ°ng output channel cá»§a `conv3` váº«n giá»¯ nguyÃªn. CÃ¡c strategy nÃ y dÃ¹ng student kiáº¿n trÃºc `middle_pruned_unet_plus_plus`.

`S9/full_static`, `S10/full_kneedle`, `S11/full_otsu` vÃ  `S12/full_gmm` cÅ©ng dÃ nh cho `TEACHER_MODEL=unet_plus_plus`, nhÆ°ng dÃ¹ng student kiáº¿n trÃºc `full_pruning_unet_plus_plus`. CÃ¡c strategy nÃ y prune full main path cá»§a bottleneck: input/output cá»§a `conv1`, `conv2`, vÃ  cáº£ `conv3 output/bn3`. Khi output block bá»‹ Ä‘á»•i channel, code sáº½ subset/cÃ i projection residual vÃ  rebuild decoder skip inputs Ä‘á»ƒ residual add vÃ  skip connection váº«n khá»›p shape.

Interface khuyáº¿n nghá»‹:

- shell ngoÃ i cÃ¹ng dÃ¹ng `PRUNE_STRATEGY=S1` Ä‘áº¿n `S12`
- Python ná»™i bá»™ dÃ¹ng `--prune_method static/kneedle/otsu/gmm/middle_static/middle_kneedle/middle_otsu/middle_gmm/full_static/full_kneedle/full_otsu/full_gmm`
- `--prune_ratio` váº«n Ä‘Æ°á»£c giá»¯ Ä‘á»ƒ backward compatibility, nhÆ°ng nÃªn dÃ¹ng `--static_prune_ratio` cho `static`

VÃ­ dá»¥ cháº¡y trá»±c tiáº¿p Python:

```bash
# S1: static pruning 50%
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S1 --static_prune_ratio 0.5

# S2: Kneedle
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S2

# S3: Otsu
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S3

# S4: GMM
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S4

# S5: middle-static pruning 50%
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S5 --static_prune_ratio 0.5

# S6: middle-kneedle pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S6

# S7: middle-otsu pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S7

# S8: middle-gmm pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S8

# S9: full-static block pruning 50%, gá»“m cáº£ conv3 output
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S9 --static_prune_ratio 0.5

# S10/S11/S12: full block dynamic pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S10
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S11
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --prune_strategy S12
```

Khi cháº¡y qua script theo dataset, set biáº¿n mÃ´i trÆ°á»ng:

```bash
# S1: static, cÃ³ pruning á»Ÿ step 3 trong 4 epoch cuá»‘i
PRUNE_STRATEGY=S1 PRUNE_RATE=0.5 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S1: static, khÃ´ng pruning á»Ÿ step 3
PRUNE_STRATEGY=S1 PRUNE_RATE=0.5 STEP3_PRUNING=0 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S2/S3/S4: dynamic, PRUNE_RATE khÃ´ng dÃ¹ng
PRUNE_STRATEGY=S2 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S3 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S4 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S5: middle-static, dÃ¹ng PRUNE_RATE giá»‘ng S1
PRUNE_STRATEGY=S5 PRUNE_RATE=0.5 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S6: middle-kneedle, PRUNE_RATE khÃ´ng dÃ¹ng
PRUNE_STRATEGY=S6 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S7/S8: middle dynamic, PRUNE_RATE khÃ´ng dÃ¹ng
PRUNE_STRATEGY=S7 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S8 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S9: full-static, dÃ¹ng PRUNE_RATE giá»‘ng S1
PRUNE_STRATEGY=S9 PRUNE_RATE=0.5 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S10/S11/S12: full dynamic, PRUNE_RATE khÃ´ng dÃ¹ng
PRUNE_STRATEGY=S10 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S11 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S12 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
```

`run_pgd_common.sh` sáº½ tá»± map strategy vÃ  táº¡o thÆ° má»¥c experiment con theo format:

```text
output_<prune_method>_<rate-or-auto>_<step3-pruning-epochs-or-no>
```

Script cháº¡y theo dataset:

```bash
bash run_pgd_cvc_clinicdb.sh
bash run_pgd_kvasir_seg.sh
bash run_pgd_etis.sh
bash run_pgd_cvc_colondb.sh
bash run_pgd_cvc_300.sh
```

Äá»•i kiáº¿n trÃºc teacher báº±ng biáº¿n mÃ´i trÆ°á»ng, vÃ­ dá»¥:

```bash
TEACHER_MODEL=unet_plus_plus PRUNE_STRATEGY=S1 bash run_pgd_etis.sh
```

| Strategy | Experiment folder |
|---|---|
| `S1`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s1_static_0.5_4` |
| `S1`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=0` | `output_s1_static_0.5_no` |
| `S2`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s2_kneedle_auto_4` |
| `S3`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s3_otsu_auto_4` |
| `S4`, `STEP3_PRUNING=0` | `output_s4_gmm_auto_no` |
| `S5`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s5_middle_static_0.5_4` |
| `S6`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s6_middle_kneedle_auto_4` |
| `S7`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s7_middle_otsu_auto_4` |
| `S8`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s8_middle_gmm_auto_4` |
| `S9`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s9_full_static_0.5_4` |
| `S10`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s10_full_kneedle_auto_4` |
| `S11`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s11_full_otsu_auto_4` |
| `S12`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_s12_full_gmm_auto_4` |

Log khi cháº¡y sáº½ cÃ³ dáº¡ng:

```text
Pruning strategy: static
Static prune ratio: 0.5
Step-3 pruning: enabled
Step-3 pruning epochs: 4
Experiment folder: output_s1_static_0.5_4
Teacher output root: outputs
Loss ablation: kd=1 sparsity=0 feat=0 aux=0
Loss weights: kd=0.3 sparsity=0.0 feat=0.1 aux=0.2
```

### Loss ablation cho Step 3

PGD student training hiá»‡n táº­p trung vÃ o 2 cháº¿ Ä‘á»™ chÃ­nh: segmentation only vÃ  segmentation + teacher output KD. Feature distillation, auxiliary loss vÃ  sparsity/gate loss váº«n cÃ²n flag tÆ°Æ¡ng thÃ­ch nhÆ°ng máº·c Ä‘á»‹nh táº¯t.

CÃ¡c argument chÃ­nh:

- `--use_kd_output 1/0`, máº·c Ä‘á»‹nh `1`
- `--use_sparsity 1/0`, máº·c Ä‘á»‹nh `0` vÃ  khÃ´ng dÃ¹ng trong pipeline chÃ­nh
- `--use_feature_distill 1/0`, máº·c Ä‘á»‹nh `0`
- `--use_aux_loss 1/0`, máº·c Ä‘á»‹nh `0`
- `--seg_loss_method ce|dice|hybrid`, máº·c Ä‘á»‹nh `hybrid`
- `--distill_loss_method mse|ce|dice|kl|hybrid`, máº·c Ä‘á»‹nh `mse`; `mse` chÃ­nh lÃ  distillation cÅ© cá»§a repo
- `--loss_method`, optional label cho Table III vÃ  output folder, vÃ­ dá»¥ `"Proposed + KL KD"`
- `--lambda_feat`, máº·c Ä‘á»‹nh `0.1`, chá»‰ cÃ³ tÃ¡c dá»¥ng khi báº­t feature distill
- `--lambda_aux`, máº·c Ä‘á»‹nh `0.2`, chá»‰ cÃ³ tÃ¡c dá»¥ng khi báº­t aux loss
- `--feature_layers`, máº·c Ä‘á»‹nh `bottleneck`

Tá»•ng loss:

```text
Ltotal = Lseg
       + lambda_distill * L_KD_output      náº¿u use_kd_output=1
```

Náº¿u loss nÃ o táº¯t, history vÃ  metrics váº«n ghi loss Ä‘Ã³ báº±ng `0.0` Ä‘á»ƒ dá»… váº½ biá»ƒu Ä‘á»“. Hai mode khuyáº¿n nghá»‹ cho paper lÃ  `loss_seg_only` vÃ  `loss_seg_kd`.

`loss_tag` Ä‘Æ°á»£c tá»± sinh theo loss Ä‘ang báº­t. CÃ¡c cáº¥u hÃ¬nh default váº«n giá»¯ tÃªn folder cÅ©, cÃ²n khi Ä‘á»•i segmentation/KD loss thÃ¬ tag sáº½ thÃªm tÃªn loss Ä‘á»ƒ phÃ¢n biá»‡t:

| Config | loss_tag |
|---|---|
| segmentation only | `loss_seg_only` |
| segmentation + KD | `loss_seg_kd` |
| segmentation + KD + feature distill | `loss_seg_kd_feat` |
| segmentation + KD + feature distill + aux | `loss_seg_kd_feat_aux` |
| CE segmentation only | `loss_seg_ce_only` |
| Dice segmentation only | `loss_seg_dice_only` |
| segmentation + KL KD | `loss_seg_kd_kl` |
| segmentation + Dice KD | `loss_seg_kd_dice` |
| custom label `Proposed + KL KD` | `loss_Proposed_KL_KD` |

VÃ¬ cÃ³ `loss_tag`, output proposal hiá»‡n cÃ³ dáº¡ng:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/<loss_tag>/<experiment_folder>/
```

VÃ­ dá»¥:

```text
outputs/pgd_unet/cvc_clinicdb/unet_plus_plus_teacher/loss_seg_kd/output_s4_gmm_auto_no/
```

Teacher váº«n dÃ¹ng chung á»Ÿ:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/
```

NghÄ©a lÃ  Ä‘á»•i ablation loss khÃ´ng lÃ m Ä‘á»•i logic pretrain/load/reuse teacher.

VÃ­ dá»¥ cháº¡y ablation qua script dataset:

```bash
# Segmentation only
USE_KD_OUTPUT=0 USE_SPARSITY=0 USE_FEATURE_DISTILL=0 USE_AUX_LOSS=0 bash run_pgd_cvc_clinicdb.sh

# Segmentation + KD
USE_KD_OUTPUT=1 USE_SPARSITY=0 USE_FEATURE_DISTILL=0 USE_AUX_LOSS=0 bash run_pgd_cvc_clinicdb.sh

# Optional legacy feature + aux study, still without sparsity/gate
USE_KD_OUTPUT=1 USE_SPARSITY=0 USE_FEATURE_DISTILL=1 USE_AUX_LOSS=1 LAMBDA_FEAT=0.1 LAMBDA_AUX=0.2 bash run_pgd_cvc_clinicdb.sh

# Table III: CE / Dice / KL / Hybrid loss study
USE_KD_OUTPUT=0 USE_SPARSITY=0 SEG_LOSS_METHOD=ce LOSS_METHOD="CE" bash run_pgd_cvc_clinicdb.sh
USE_KD_OUTPUT=0 USE_SPARSITY=0 SEG_LOSS_METHOD=dice LOSS_METHOD="Dice" bash run_pgd_cvc_clinicdb.sh
DISTILL_LOSS_METHOD=kl LOSS_METHOD="Proposed + KL KD" bash run_pgd_cvc_clinicdb.sh
DISTILL_LOSS_METHOD=hybrid LOSS_METHOD="Proposed + Hybrid KD" bash run_pgd_cvc_clinicdb.sh
```

`TEACHER_OUTPUT_ROOT` dÃ¹ng Ä‘á»ƒ cá»‘ Ä‘á»‹nh nÆ¡i lÆ°u `1_teacher`. Máº·c Ä‘á»‹nh cÃ¡c script dataset dÃ¹ng `outputs`, nÃªn teacher chá»‰ cáº§n train má»™t láº§n táº¡i:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/
```

Náº¿u trÆ°á»›c Ä‘Ã³ teacher Ä‘Ã£ Ä‘Æ°á»£c train trong má»™t output thá»­ nghiá»‡m cÅ©, pipeline sáº½ cá»‘ gáº¯ng tÃ¬m checkpoint teacher compatible á»Ÿ output hiá»‡n táº¡i vÃ  register/copy nÃ³ sang `TEACHER_OUTPUT_ROOT` trÆ°á»›c khi quyáº¿t Ä‘á»‹nh train teacher láº¡i.

CÃ¡c láº§n thá»­ pruning khÃ¡c nhau váº«n ghi vÃ o folder con bÃªn trong cÃ¹ng root teacher, vÃ­ dá»¥:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s1_static_0.5_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s1_static_0.5_no/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s2_kneedle_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s5_middle_static_0.5_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s6_middle_kneedle_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s7_middle_otsu_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_s8_middle_gmm_auto_4/2_pruning/
```

### Teacher reuse

Khi cáº§n teacher checkpoint, `train_pgd.py` sáº½ check theo thá»© tá»±:

1. explicit `--teacher_checkpoint` náº¿u cÃ³ truyá»n vÃ o
2. `outputs/pgd_unet/<dataset>/<teacher>_teacher/1_teacher/checkpoints/`
3. `outputs/<teacher_model>/<dataset>/checkpoints/`

Náº¿u reuse tá»« `basic branch`, checkpoint sáº½ Ä‘Æ°á»£c register láº¡i vÃ o `1_teacher/checkpoints/` Ä‘á»ƒ output proposal luÃ´n Ä‘áº§y Ä‘á»§ vÃ  dá»… Ä‘á»c.

Máº·c Ä‘á»‹nh repo **khÃ´ng train láº¡i teacher náº¿u Ä‘Ã£ cÃ³ checkpoint compatible**. Chá»‰ khi cáº£ 3 nguá»“n trÃªn Ä‘á»u khÃ´ng cÃ³ checkpoint phÃ¹ há»£p, hoáº·c báº¡n báº­t `--force_retrain_teacher 1`, thÃ¬ `1_teacher` má»›i train láº¡i tá»« Ä‘áº§u.

Náº¿u teacher pháº£i train láº¡i tá»« Ä‘áº§u vÃ  `TEACHER_MODEL=unet_plus_plus`, máº·c Ä‘á»‹nh repo sáº½ dÃ¹ng `encoder_pretrained=1`.

### CÃ¡ch táº­n dá»¥ng weight teacher Ä‘á»ƒ khÃ´ng cáº§n train láº¡i

CÃ³ 2 cÃ¡ch dÃ¹ng thá»±c táº¿:

#### CÃ¡ch 1: train basic model trÆ°á»›c, rá»“i Ä‘á»ƒ proposal tá»± reuse

VÃ­ dá»¥ train baseline teacher trÆ°á»›c:

```bash
python train_basic_model.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

Sau Ä‘Ã³ cháº¡y proposal vá»›i cÃ¹ng `teacher_model` vÃ  `dataset`:

```bash
python train_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --encoder_pretrained 1 --max_epochs_teacher 50 --max_epochs_student 100 --prune_strategy S1 --static_prune_ratio 0.5
```

Khi Ä‘Ã³ `train_pgd.py` sáº½:

- tÃ¬m teacher checkpoint trong `outputs/unet_resnet152/cvc_clinicdb/checkpoints/`
- load láº¡i weight Ä‘Ã³
- register/copy checkpoint vÃ o `outputs/pgd_unet/cvc_clinicdb/unet_plus_plus_teacher/1_teacher/checkpoints/`
- skip teacher training náº¿u checkpoint compatible

#### CÃ¡ch 2: chá»‰ Ä‘á»‹nh trá»±c tiáº¿p teacher checkpoint

Báº¡n cÅ©ng cÃ³ thá»ƒ chá»‰ Ä‘á»‹nh tháº³ng má»™t checkpoint Ä‘Ã£ cÃ³:

```bash
python train_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_plus_plus --teacher_checkpoint outputs/unet_resnet152/cvc_clinicdb/checkpoints/best.pth --encoder_pretrained 1 --max_epochs_student 100 --prune_strategy S1 --static_prune_ratio 0.5
```

Repo sáº½ kiá»ƒm tra compatibility cá»§a checkpoint nÃ y trÆ°á»›c khi dÃ¹ng. Náº¿u checkpoint nÃ y tá»“n táº¡i vÃ  compatible, nÃ³ sáº½ Ä‘Æ°á»£c Æ°u tiÃªn dÃ¹ng trÆ°á»›c proposal outputs vÃ  basic outputs. Náº¿u checkpoint khÃ´ng tá»“n táº¡i hoáº·c khÃ´ng tÆ°Æ¡ng thÃ­ch, pipeline má»›i tiáº¿p tá»¥c check cÃ¡c nguá»“n cÃ²n láº¡i.

Khi `--teacher_checkpoint` há»£p lá»‡, pipeline sáº½:

- load trá»±c tiáº¿p weight tá»« Ä‘Ãºng path báº¡n truyá»n vÃ o
- sau Ä‘Ã³ register/mirror checkpoint Ä‘Ã³ vÃ o `outputs/pgd_unet/.../1_teacher/checkpoints/`
- dÃ¹ng báº£n register nÃ y nhÆ° proposal-owned checkpoint Ä‘á»ƒ export artifact vÃ  dá»… tÃ¡i sá»­ dá»¥ng vá» sau

#### Khi nÃ o proposal sáº½ train láº¡i teacher?

Teacher chá»‰ train láº¡i khi:

- khÃ´ng cÃ³ checkpoint compatible trong proposal outputs
- khÃ´ng cÃ³ checkpoint compatible tá»« `--teacher_checkpoint`
- khÃ´ng cÃ³ checkpoint compatible trong basic outputs
- hoáº·c báº¡n chá»§ Ä‘á»™ng báº­t `--force_retrain_teacher 1`

### Pipeline phases

Pipeline proposal gá»“m:

1. `1_teacher`
2. `2_pruning`
3. `3_student`
4. `student_final`
5. `pipeline`

### Step 2: Pruning / pruned-student initialization

Step 2 hiá»‡n khÃ´ng chá»‰ sinh `blueprint` rá»“i dá»±ng má»™t student má»›i hoÃ n toÃ n.

Luá»“ng hiá»‡n táº¡i lÃ :

1. phÃ¢n tÃ­ch teacher vÃ  chá»n `kept_channel_indices` theo `prune_method`
2. sinh `blueprint` vÃ  `channel_config` cho student
3. build `pruned student` tá»« blueprint: S1-S4 dÃ¹ng `blueprint_unet_plus_plus` khi teacher lÃ  `unet_plus_plus`, S5-S8 dÃ¹ng `middle_pruned_unet_plus_plus`, S9-S12 dÃ¹ng `full_pruning_unet_plus_plus`
4. náº¡p láº¡i weight cá»§a teacher á»©ng vá»›i cÃ¡c channel Ä‘Æ°á»£c giá»¯ láº¡i vÃ o tá»«ng stage encoder cá»§a student
5. copy thÃªm cÃ¡c tensor cÃ²n tÆ°Æ¡ng thÃ­ch báº±ng exact-match fallback
6. evaluate `train / val / test`

Äiá»u nÃ y cÃ³ nghÄ©a:

- `2_pruning` khÃ´ng cÃ²n lÃ  `random init baseline`
- student á»Ÿ bÆ°á»›c nÃ y Ä‘Ã£ reuse láº¡i pháº§n weight sá»‘ng sÃ³t tá»« teacher whenever compatible
- metric á»Ÿ `2_pruning` vÃ¬ váº­y pháº£n Ã¡nh `pruned student after structural pruning + teacher weight subset reuse`, chá»© khÃ´ng pháº£i má»™t model má»›i hoÃ n toÃ n tá»« Ä‘áº§u
- `blueprint.json` lÆ°u thÃªm `prune_strategy`, `prune_method`, `static_prune_ratio`, `pruning_threshold` vÃ  `kept_channel_indices` Ä‘á»ƒ trace láº¡i strategy Ä‘Ã£ dÃ¹ng

### Step 2 weight-transfer logging

Khi cháº¡y `2_pruning`, repo sáº½ log rÃµ cho tá»«ng stage:

- thÃ nh pháº§n nÃ o `copy trá»±c tiáº¿p`
- thÃ nh pháº§n nÃ o `resize kernel rá»“i copy`
- thÃ nh pháº§n nÃ o `khÃ´ng map Ä‘Æ°á»£c`

VÃ­ dá»¥ log:

```text
Step-2 stage reuse | student_stage=stem | teacher_module=stem | status=channel_subset_reused | direct=first_norm,second_conv,second_norm | resized=first_conv | unmapped=none
Step-2 stage reuse | student_stage=down1 | teacher_module=layer1 | status=channel_subset_reused | direct=first_conv,first_norm | resized=none | unmapped=second_conv,second_norm
Step-2 head reuse | status=copied | mode=direct | reason=
```

Metadata nÃ y cÅ©ng Ä‘Æ°á»£c lÆ°u trong `weight_transfer` cá»§a phase pruning Ä‘á»ƒ cÃ³ thá»ƒ xem láº¡i sau khi train:

- `stage_transfer_rows[*].direct_components`
- `stage_transfer_rows[*].resized_components`
- `stage_transfer_rows[*].unmapped_components`
- `head_transfer`
- `exact_match_copy_ratio`

## 7. Step 3: Student training / compression training

Step 3 hiá»‡n bÃ¡m theo logic implementation sau:

1. load blueprint vÃ  pruned student checkpoint tá»« `2_pruning`
2. build láº¡i Ä‘Ãºng student architecture tá»« blueprint (`blueprint_unet_plus_plus`, `middle_pruned_unet_plus_plus`, `full_pruning_unet_plus_plus`, hoáº·c fallback `PDGUNet`)
3. load teacher Ä‘Ã£ freeze tá»« `1_teacher`
4. train student vá»›i:
   `Ltotal = Lseg` hoáº·c `Ltotal = Lseg + lambda_distill * Ldistill`

Student á»Ÿ Ä‘áº§u step 3 khÃ´ng khá»Ÿi táº¡o random:

- weight ban Ä‘áº§u Ä‘Æ°á»£c load tá»« checkpoint cá»§a `2_pruning`
- checkpoint nÃ y Ä‘Ã£ chá»©a pruned-student sau bÆ°á»›c structural pruning vÃ  teacher-weight subset reuse á»Ÿ step 2

Pipeline chÃ­nh khÃ´ng dÃ¹ng gate sparsity hoáº·c late hard pruning á»Ÿ step 3 ná»¯a. `step3_pruning_epochs`, `warmup_pruning_epochs` vÃ  cÃ¡c method gate cÃ²n tá»“n táº¡i Ä‘á»ƒ khÃ´ng phÃ¡ checkpoint/script cÅ©, nhÆ°ng default path chá»‰ fine-tune student bÃ¬nh thÆ°á»ng, cÃ³ hoáº·c khÃ´ng cÃ³ KD.

Trong `3_student/configs/`, repo hiá»‡n lÆ°u thÃªm `student_pruning_config.json` Ä‘á»ƒ nhÃ¬n nhanh cÃ¡c má»‘c epoch cá»§a pruning, vÃ­ dá»¥:

- `requested_warmup_pruning_epochs`
- `warmup_pruning_epochs`
- `step3_pruning_enabled`
- `step3_pruning_epochs`
- `hard_pruning_apply_epoch`
- `hard_pruning_apply_epoch_0based`
- `hard_pruning_apply_epoch_1based`
- `hard_pruning_start_epoch`
- `late_pruning_epoch_window`
- `late_pruning_epoch_window_0based`
- `gate_search_epoch_window`
- `gate_search_epoch_window_0based`
- `effective_late_pruning_epochs`

CÃ¡c field nÃ y Ä‘Æ°á»£c giá»¯ chá»§ yáº¿u Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch cáº¥u hÃ¬nh cÅ©:

- `step3_pruning_enabled` cho biáº¿t step 3 cÃ³ báº­t pruning legacy hay khÃ´ng
- `step3_pruning_epochs` lÃ  sá»‘ epoch cuá»‘i dÃ nh cho late pruning legacy náº¿u ngÆ°á»i dÃ¹ng tá»± báº­t
- `requested_warmup_pruning_epochs` lÃ  giÃ¡ trá»‹ báº¡n truyá»n tá»« CLI hoáº·c giÃ¡ trá»‹ Ä‘Æ°á»£c map tá»« `step3_pruning_epochs`
- `warmup_pruning_epochs` lÃ  giÃ¡ trá»‹ effective tháº­t sau khi Ä‘Ã£ clamp theo `max_epochs_student`
- `hard_pruning_apply_epoch` lÃ  má»‘c legacy theo **0-based index**
- vá»›i pipeline chÃ­nh hiá»‡n táº¡i, cÃ¡c giÃ¡ trá»‹ nÃ y thÆ°á»ng báº±ng tráº¡ng thÃ¡i táº¯t/khÃ´ng dÃ¹ng

### Distillation

Hiá»‡n táº¡i distillation target lÃ  `logits`, chÆ°a pháº£i feature adapter riÃªng.

### Gate compatibility

Gate/gated UNet khÃ´ng cÃ²n lÃ  pipeline chÃ­nh. `PDGUNet` bÃ¢y giá» lÃ  pruned UNet bÃ¬nh thÆ°á»ng; cÃ¡c method nhÆ° `get_gate_tensors()`, `force_gates_open()` vÃ  `set_gate_trainable()` chá»‰ cÃ²n lÃ  no-op Ä‘á»ƒ checkpoint/script cÅ© khÃ´ng vá»¡.

Flag `--enable_step3_pruning`, `--warmup_pruning_epochs`, `--lambda_sparsity`, `--student_gate_near_off_threshold` vÃ  `--student_hard_gate_threshold` Ä‘Æ°á»£c giá»¯ Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch CLI cÅ©, nhÆ°ng máº·c Ä‘á»‹nh khÃ´ng kÃ­ch hoáº¡t sparsity/gate loss.

### Student variants

Step 3 nÃªn dÃ¹ng 2 mode chÃ­nh:

- `--use_kd_output 0`: segmentation only
- `--use_kd_output 1`: segmentation + KD/distillation

`student_variant=full` vÃ  `student_variant=pruned_distill` Ä‘á»u Ä‘i qua student khÃ´ng gate. `student_variant=pruned_gate_sparsity` chá»‰ cÃ²n lÃ  alias tÆ°Æ¡ng thÃ­ch cÅ© vÃ  cÅ©ng bá»‹ Ã©p táº¯t gate/sparsity trong policy hiá»‡n táº¡i.

## 8. Checkpoint vÃ  metadata

Checkpoint Ä‘Æ°á»£c quáº£n lÃ½ qua `utils/checkpoints.py`.

Má»—i checkpoint cÃ³ thá»ƒ lÆ°u:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `global_step`
- `best_metric`
- `metrics`
- `config`
- `model_info`
- `phase`
- `extra_state`

Vá»›i proposal branch, `model_info` hiá»‡n cÅ©ng ghi rÃµ checkpoint cÃ³ pháº£i random init hay khÃ´ng, vÃ­ dá»¥:

- `checkpoint_is_random_init: false`
- `checkpoint_weight_status`
- `checkpoint_weight_source`

Äiá»u nÃ y Ä‘áº·c biá»‡t há»¯u Ã­ch Ä‘á»ƒ kiá»ƒm tra:

- `2_pruning` Ä‘ang lÆ°u model Ä‘Ã£ reuse weight tá»« teacher
- `3_student` Ä‘ang lÆ°u model load tá»« `2_pruning`
- `3_student` lÆ°u metric/metadata fine-tune chÃ­nh, cÃ³ thá»ƒ lÃ  segmentation only hoáº·c segmentation + KD

Máº·c Ä‘á»‹nh lÆ°u bá»™ checkpoint gá»n cho evaluate/reuse:

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `checkpoints/config.yaml`
- `checkpoints/metrics.json`
- `checkpoints/train_log.csv`
- `checkpoints/metadata/best.json`

Má»—i phase train/evaluate cá»§a proposal cÅ©ng cÃ³ `log.txt` ngay trong thÆ° má»¥c phase, vÃ­ dá»¥:

```text
1_teacher/log.txt
2_pruning/log.txt
3_student/log.txt
pipeline/log.txt
```

File nÃ y ghi láº¡i tiáº¿n trÃ¬nh cháº¡y, thÃ´ng tin reuse checkpoint, pruning search, epoch loss/metric vÃ  cÃ¡c bÆ°á»›c export artifact. `pipeline/run.log` váº«n Ä‘Æ°á»£c giá»¯ Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch output cÅ©.

Checkpoint máº·c Ä‘á»‹nh gá»“m `model_state_dict` vÃ  metadata cáº§n thiáº¿t (`epoch`, `best_metric`, `metrics`, `config`, `model_info`, `phase`, `extra_state`). Repo khÃ´ng lÆ°u `optimizer_state_dict`/scheduler/scaler máº·c Ä‘á»‹nh Ä‘á»ƒ trÃ¡nh file quÃ¡ náº·ng.

Náº¿u muá»‘n giá»¯ thÃªm checkpoint history:

```bash
--save_history_checkpoints 1
```

Náº¿u muá»‘n táº¯t checkpoint cuá»‘i cÃ¹ng bá»‹ overwrite sau má»—i láº§n eval:

```bash
--save_last_checkpoint 0
```

Náº¿u tháº­t sá»± cáº§n resume Ä‘Ãºng optimizer state:

```bash
--save_optimizer_state 1
```

## 9. Output structure

### 9.1 Basic branch

```text
outputs/<model_name>/<dataset>/
â”œâ”€ artifacts/
â”‚  â”œâ”€ channel_analysis/
â”‚  â””â”€ visualizations/
â”œâ”€ checkpoints/
â”‚  â”œâ”€ best.pth
â”‚  â”œâ”€ last.pth
â”‚  â”œâ”€ config.yaml
â”‚  â”œâ”€ metrics.json
â”‚  â”œâ”€ train_log.csv
â”‚  â””â”€ metadata/
â”œâ”€ configs/
â”‚  â”œâ”€ run_config.json
â”‚  â”œâ”€ hyperparameters.json
â”‚  â””â”€ model_config.json
â”œâ”€ evaluations/
â”‚  â”œâ”€ train/
â”‚  â”œâ”€ val/
â”‚  â””â”€ test/
â”œâ”€ metrics/
â”‚  â””â”€ basic_metrics.csv
â”œâ”€ reports/
â”‚  â”œâ”€ basic_loss.pdf
â”‚  â”œâ”€ basic_performance.pdf
â”‚  â”œâ”€ basic_train_visualizations.pdf
â”‚  â”œâ”€ basic_val_visualizations.pdf
â”‚  â””â”€ basic_test_visualizations.pdf
â””â”€ run.log
```

### 9.2 Proposal branch

Náº¿u khÃ´ng truyá»n `--output_root`, output máº·c Ä‘á»‹nh náº±m dÆ°á»›i `outputs/`. Náº¿u cháº¡y qua script dataset, `--output_root` lÃ  root chung, máº·c Ä‘á»‹nh `outputs`; cáº¥u hÃ¬nh strategy vÃ  step 3 náº±m á»Ÿ folder con nhÆ° `output_s1_static_0.5_4`, `output_s1_static_0.5_no`, `output_s2_kneedle_auto_4`, `output_s5_middle_static_0.5_4`, `output_s6_middle_kneedle_auto_4`, `output_s7_middle_otsu_auto_4`, `output_s8_middle_gmm_auto_4`, hoáº·c `output_s4_gmm_auto_no`.

Teacher cÃ³ root riÃªng qua `--teacher_output_root`. Khi dÃ¹ng script dataset, giÃ¡ trá»‹ máº·c Ä‘á»‹nh lÃ  `outputs`, vÃ¬ váº­y:

- `1_teacher` dÃ¹ng chung: `outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/`
- `2_pruning`, `3_student`, `pipeline` theo tá»«ng thá»­ nghiá»‡m: `<output_root>/pgd_unet/<dataset>/<teacher_model>_teacher/<experiment_folder>/...`

```text
<output_root>/pgd_unet/<dataset>/<teacher_model>_teacher/
â”œâ”€ 1_teacher/
â”œâ”€ output_s1_static_0.5_4/
â”‚  â”œâ”€ 2_pruning/
â”‚  â”œâ”€ 3_student/
â”‚  â”œâ”€ student_final/
â”‚  â””â”€ pipeline/
â””â”€ output_s2_kneedle_auto_4/
   â”œâ”€ 2_pruning/
   â”œâ”€ 3_student/
   â”œâ”€ student_final/
   â””â”€ pipeline/
â””â”€ output_s5_middle_static_0.5_4/
   â”œâ”€ 2_pruning/
   â”œâ”€ 3_student/
   â”œâ”€ student_final/
   â””â”€ pipeline/
â””â”€ output_s6_middle_kneedle_auto_4/
   â”œâ”€ 2_pruning/
   â”œâ”€ 3_student/
   â”œâ”€ student_final/
   â””â”€ pipeline/
â””â”€ output_s7_middle_otsu_auto_4/
   â”œâ”€ 2_pruning/
   â”œâ”€ 3_student/
   â”œâ”€ student_final/
   â””â”€ pipeline/
â””â”€ output_s8_middle_gmm_auto_4/
   â”œâ”€ 2_pruning/
   â”œâ”€ 3_student/
   â”œâ”€ student_final/
   â””â”€ pipeline/
```

#### `1_teacher`

LÆ°u:

- checkpoint teacher
- configs
- metrics
- reports
- evaluations
- channel analysis
- visualizations

#### `2_pruning`

LÆ°u:

- `blueprint.json`
- pruning summary
- teacher vs student channel comparison
- pruning analysis
- pruned-student baseline checkpoint
- pruned-student evaluation `train/val/test`
- metadata vá» `weight_transfer`

Quan trá»ng:

- `2_pruning` bÃ¢y giá» khÃ´ng chá»‰ lÃ  stage kiáº¿n trÃºc
- nÃ³ cÃ²n evaluate má»™t `pruned student before tuning` Ä‘á»ƒ báº¡n nhÃ¬n Ä‘Æ°á»£c hiá»‡u quáº£ ngay sau pruning
- pruned student nÃ y Ä‘Æ°á»£c khá»Ÿi táº¡o báº±ng cÃ¡c weight cá»§a teacher tÆ°Æ¡ng á»©ng vá»›i nhá»¯ng channel khÃ´ng bá»‹ prune, thay vÃ¬ Ä‘á»ƒ toÃ n bá»™ trá»ng sá»‘ á»Ÿ tráº¡ng thÃ¡i khá»Ÿi táº¡o má»›i
- `weight_transfer` hiá»‡n cÅ©ng ghi rÃµ stage nÃ o `direct copy`, stage nÃ o `resized kernel`, stage nÃ o `unmapped`, vÃ  `head` cÃ³ reuse Ä‘Æ°á»£c hay khÃ´ng

#### `3_student`

LÆ°u:

- checkpoint student
- configs
- `configs/student_pruning_config.json`
- metrics
- reports
- evaluations
- `artifacts/channel_analysis/`
- `artifacts/visualizations/`
- `artifacts/gating_analysis/`
- `artifacts/student_tuning_analysis/`
- `metrics/student_epoch_diagnostics.csv`
- `reports/student_channel_gating_report_tables.pdf`
- `reports/student_channel_gating_report_charts.pdf`

#### `student_final`

Shortcut export cho weight cuá»‘i:

```text
student_final/
â”œâ”€ best_student.pth
â”œâ”€ last_student.pth
â”œâ”€ best_student.json
â””â”€ last_student.json
```

#### `pipeline`

LÆ°u report tá»•ng há»£p toÃ n pipeline:

- `metrics/pipeline_stage_overview.csv`
- `metrics/pipeline_metrics.csv`
- `metrics/pipeline_compression_summary.csv`
- `reports/pipeline_performance.pdf`
- `reports/pipeline_<phase>_<split>_visualizations.pdf`
- `evaluations/pipeline_summary.json`
- `evaluations/pipeline_summary.md`

Pipeline tá»•ng há»£p cáº£:

- teacher metrics
- pruning baseline metrics
- final student metrics
- compression summary

## 10. Evaluation vÃ  metrics

Metric chÃ­nh:

- Dice
- IoU
- HD95

ThÃ´ng tin efficiency:

- Params
- FLOPs
- FPS
- inference time

Má»—i phase cÃ³ thá»ƒ export:

- `case_metrics.csv`
- `summary.json`
- `summary.md`
- `metrics_summary.json`

CÃ¡c split cuá»‘i máº·c Ä‘á»‹nh:

- `train`
- `val`
- `test`

## 11. Reports vÃ  artifacts

### PDF reports

Repo hiá»‡n há»— trá»£:

- `loss.pdf`
- `visualizations.pdf`
- `performance.pdf`
- `channel_analysis_tables.pdf`
- `channel_analysis_charts.pdf`
- `student_channel_report_tables.pdf`
- `student_channel_report_charts.pdf`

### Channel / pruning artifacts

CÃ¡c nhÃ³m artifact chÃ­nh:

- `artifacts/channel_analysis/`
- `artifacts/visualizations/`
- `artifacts/student_tuning_analysis/`
- `artifacts/pruning_analysis/`

### Student step-3 diagnostics

Step 3 export thÃªm:

- student input channel profile
- student final channel profile
- so sÃ¡nh `student_input -> student_final`
- per-epoch diagnostics vÃ  `checkpoints/train_log.csv` Ä‘á»ƒ váº½ validation curves

## 12. Má»™t sá»‘ lÆ°u Ã½ thá»±c táº¿

- `net_factory.py` khÃ´ng route `pdg_unet`; Ä‘Ã¢y lÃ  chá»§ Ä‘Ã­ch Ä‘á»ƒ tÃ¡ch `basic` vÃ  `proposal`
- `thop` cÃ³ thá»ƒ gáº¯n `total_ops/total_params` vÃ o model; repo Ä‘Ã£ strip cÃ¡c key nÃ y khi save/load checkpoint
- path trong metadata Æ°u tiÃªn dáº¡ng tÆ°Æ¡ng Ä‘á»‘i nhÆ° `outputs/...` hoáº·c `evaluations/...`, khÃ´ng lÆ°u absolute path
- `test2d.py` Æ°u tiÃªn output má»›i trong `outputs/...`, nhÆ°ng váº«n fallback Ä‘Æ°á»£c cho run legacy náº¿u cáº§n
- distillation cá»§a step 3 hiá»‡n lÃ  `logits distillation`
- gate/sparsity khÃ´ng cÃ²n thuá»™c pipeline chÃ­nh; code gate cÅ© chá»‰ cÃ²n compatibility no-op

## 13. compare_artifacts.py

Repo cÃ³ thÃªm `compare_artifacts.py` Ä‘á»ƒ gom káº¿t quáº£:

- `basic baseline`
- `teacher`
- `pruned student`
- `tuned student`

thÃ nh má»™t report thá»‘ng nháº¥t.

### VÃ­ dá»¥ dÃ¹ng pipeline summary

```bash
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --pipeline_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/pipeline --comparison_name <report_name>
```

### VÃ­ dá»¥ chá»‰ Ä‘á»‹nh tá»«ng phase

```bash
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --teacher_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher --pruning_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/2_pruning --student_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/3_student --output_dir outputs/comparisons/<report_name>
```

Script sáº½ sinh:

- `stage_overview.csv`
- `performance_comparison.csv`
- `pruning_global_summary.csv`
- `teacher_vs_student_channels.csv`
- `student_tuning_comparison.csv`
- `comparison_summary.json`
- `<comparison_name>.pdf`

## 14. Quick start

1. CÃ i dependency
2. Chuáº©n bá»‹ dataset vÃ  split
3. Train baseline báº±ng `train_basic_model.py`
4. Train proposal báº±ng `train_pgd.py`
5. Äá»c output trong `outputs/`
6. So sÃ¡nh káº¿t quáº£ báº±ng `compare_artifacts.py`

## 15. Paper experiments and statistics

### How to run basic models

Cháº¡y baseline/basic models trÃªn nhiá»u dataset báº±ng wrapper á»Ÿ root project:

```bash
bash run_basic_models.sh
```

CÃ³ thá»ƒ chá»‰nh nhanh báº±ng biáº¿n mÃ´i trÆ°á»ng:

```bash
DATASETS="cvc_300 cvc_clinicdb kvasir_seg etis cvc_colondb" MODELS="unet resunet vnet unetr unet_resnet152 att_unet r2unet unet_plus_plus" EPOCHS=50 BATCH_SIZE=8 bash run_basic_models.sh
```

Hoáº·c gá»i Python wrapper trá»±c tiáº¿p:

```bash
python run_basic_model.py --datasets cvc_300 kvasir_seg --models unet resunet --output-root outputs --device 0 --epochs 50 --batch-size 8
```

### Script folders

- `scripts_pgd_s1_s12_student_losses/`: PGD S1-S12, tach rieng script `seg_only` va `seg_kd` cho moi dataset.
- `scripts_basic_models/`: basic models, mac dinh gom `unet`, `resunet`, `vnet`, `unetr`, `unet_resnet152`, `att_unet`, `r2unet`, `unet_plus_plus`.

Chay PGD theo tung loss:

```bash
bash scripts_pgd_s1_s12_student_losses/run_pgd_s1_s12_kvasir_seg_seg_only.sh
bash scripts_pgd_s1_s12_student_losses/run_pgd_s1_s12_kvasir_seg_seg_kd.sh
```

Co the chia S-runs hoac basic models len nhieu GPU bang `DEVICE`:

```bash
DEVICE="0 1" bash scripts_pgd_s1_s12_student_losses/run_pgd_s1_s12_kvasir_seg_seg_kd.sh
DEVICE="0 1" bash scripts_basic_models/run_basic_models_kvasir_seg.sh
```

Voi PGD, run dau tien se chay tuan tu de tao/reuse teacher checkpoint, sau do cac S tiep theo moi chay song song theo danh sach GPU. Neu teacher da co san, co the bo buoc nay:

```bash
DEVICE="0 1" PGD_TEACHER_PREPARE_FIRST=0 bash scripts_pgd_s1_s12_student_losses/run_pgd_s1_s12_kvasir_seg_seg_kd.sh
```

Wrapper nÃ y gá»i láº¡i `train_basic_model.py`, nÃªn output váº«n náº±m trong cáº¥u trÃºc cÅ©:

```text
outputs/<model>/<dataset>/
```

### How to run pruning experiments

CÃ¡c pruning/proposal experiments váº«n cháº¡y báº±ng cÃ¡c script root hiá»‡n cÃ³:

```bash
bash run_pgd_cvc_300.sh
bash run_pgd_cvc_clinicdb.sh
bash run_pgd_kvasir_seg.sh
bash run_pgd_etis.sh
bash run_pgd_cvc_colondb.sh
```

Chá»n pruning strategy báº±ng biáº¿n mÃ´i trÆ°á»ng:

```bash
PRUNE_STRATEGY=S2 bash run_pgd_cvc_300.sh
PRUNE_STRATEGY=S5 PRUNE_RATE=0.5 bash run_pgd_cvc_300.sh
```

Táº¯t distillation cho ablation:

```bash
USE_KD_OUTPUT=0 LAMBDA_DISTILL=0 bash run_pgd_cvc_300.sh
```

### Extra run artifacts for paper

Má»—i run giá»¯ nguyÃªn output cÅ© vÃ  cÃ³ thá»ƒ cÃ³ thÃªm sidecar:

```text
metrics_summary.csv
sample_metrics.csv
timing_summary.json
timing_summary.csv
pruning_search_time.json
inference_summary.csv
```

`sample_metrics.csv` dÃ¹ng Ä‘á»ƒ sort cÃ¡c case Dice tháº¥p nháº¥t:

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/.../evaluations/test/sample_metrics.csv'); print(df.sort_values('dice').head(10))"
```

### How to run statistics

Statistics chá»‰ Ä‘á»c `outputs/`, khÃ´ng train láº¡i:

```bash
bash run_statistics.sh --outputs-root outputs --save-root statistics/outputs --dataset-main cvc_300
```

Lá»‡nh nÃ y cháº¡y:

```text
statistics/src/generate_tables.py
statistics/src/generate_figures.py
statistics/src/collect_paper_artifacts.py
```

### Where paper-ready tables are saved

Vá»›i má»—i dataset:

```text
statistics/outputs/<dataset_name>/table1_baseline.csv
statistics/outputs/<dataset_name>/table2_pruning.csv
statistics/outputs/<dataset_name>/table3_loss.csv
statistics/outputs/<dataset_name>/table4_ablation.csv
statistics/outputs/<dataset_name>/table5_computational_cost.csv
```

Figure 5 duoc xuat theo ca file tong hop va 3 file rieng theo nhom pruning:

```text
statistics/outputs/<dataset_name>/figure5_layerwise_pruning_ratio.pdf
statistics/outputs/<dataset_name>/figure5_s1_s4_blueprint_stage_pruning_ratio.pdf
statistics/outputs/<dataset_name>/figure5_s5_s8_middle_conv2_layerwise_pruning_ratio.pdf
statistics/outputs/<dataset_name>/figure5_s9_s12_full_block_layerwise_pruning_ratio.pdf
```

Phase `2_pruning` cung xuat them bang/chung cu copy weight:

```text
2_pruning/artifacts/weight_transfer/weight_transfer.json
2_pruning/artifacts/weight_transfer/stage_transfer_rows.csv
2_pruning/artifacts/weight_transfer/block_transfer_rows.csv
2_pruning/artifacts/weight_transfer/decoder_subset_transfer_rows.csv
```

Báº£ng mean/std across datasets:

```text
statistics/outputs/table_mean_std_across_datasets.csv
```

### Where paper-ready figures are saved

Figures theo dataset:

```text
statistics/outputs/<dataset_name>/figure*.pdf
```

Figures chung:

```text
statistics/outputs/paper_figures/
statistics/outputs/figure15_mean_performance_across_datasets.pdf
```

Paper-ready copies:

```text
statistics/paper_ready/main/
statistics/paper_ready/appendix/
statistics/paper_ready/manifest.csv
```

`--dataset-main` chá»n dataset Æ°u tiÃªn cho main paper. CÃ¡c dataset cÃ²n láº¡i váº«n cÃ³ báº£ng/hÃ¬nh trong `statistics/outputs/<dataset_name>/` Ä‘á»ƒ dÃ¹ng cho appendix hoáº·c phÃ¢n tÃ­ch bá»• sung.
