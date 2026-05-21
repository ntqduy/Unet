# Tong Hop Codebase PGD-UNet

File nay tom tat nhung gi code hien tai dang lam trong repo. Muc tieu chinh cua repo la huan luyen va danh gia bai toan medical image segmentation, dong thoi thuc hien pruning, compression va knowledge distillation cho mo hinh UNet/ResNet-UNet tren cac dataset polyp.

## 1. Muc Tieu Tong Quan

Codebase dang benchmark hai nhanh chinh:
Code ho tro cac dataset polyp:

- `kvasir_seg`
- `cvc_clinicdb`
- `cvc_colondb`
- `cvc_300`
- `etis`

Dataset duoc doc qua `dataloaders/dataset.py`. Mask duoc normalize ve binary label `0/1`. Cac split `train`, `val`, `test` duoc luu trong:

```text
data/<dataset>/splits/
```

Voi Kvasir-SEG, code dung manifest goc roi chia on dinh train/val/test. Voi cac dataset khac, code co script tao split co dinh de tranh moi lan train bi chia khac nhau.

## 3. Basic Branch

Basic branch train cac model segmentation rieng le de lam baseline.

Model hien co trong `networks/net_factory.py`:

- `unet`: U-Net 2D co ban.
- `resunet`: Residual U-Net.
- `vnet`: V-Net 2D.
- `unetr`: UNETR 2D.
- `unet_resnet152`: U-Net decoder voi ResNet152 encoder.
- `att_unet`: Attention U-Net tu reference `src_ref/Image_Segmentation`.
- `r2unet`: R2U-Net/Recurrent Residual U-Net tu reference `src_ref/Image_Segmentation`.
- `unet_plus_plus`: UNet++ wrapper dua tren `segmentation_models_pytorch`.

Vi du:

```bash
python train_basic_model.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --max_epochs 100
```

Output cua basic branch nam o:

 `networks/PGD_Unet/middle_pruned_unet_plus_plus.py`
 `networks/PGD_Unet/full_pruning_unet_plus_plus.py`
```

Moi run co checkpoint, metrics, log, reports va visualization neu duoc bat.

## 4. Proposal Branch: PGD-UNet

Proposal branch co cac phase:

```text
1_teacher/
2_pruning/
3_student/
student_final/
pipeline/
```

Output co dang:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/<loss_tag>/<output_dir>/
```

Vi du:

```text
outputs/pgd_unet/kvasir_seg/unet_resnet152_teacher/loss_seg_kd/output_otsu_auto_no/
```

## 5. Phase 1: Teacher

Phase `1_teacher` train hoac reuse mot teacher model lon, thuong la:

```text
unet_resnet152
```

Code uu tien load teacher checkpoint theo thu tu:

1. `--teacher_checkpoint` neu duoc chi dinh.
2. Checkpoint trong proposal output `1_teacher`.
3. Checkpoint trong basic branch `outputs/<teacher_model>/<dataset>/checkpoints/`.

Neu khong co checkpoint phu hop, code moi train teacher tu dau.

## 6. Phase 2: Pruning

Phase `2_pruning` phan tich importance cua channel trong teacher, chon channel giu lai, tao `blueprint.json`, roi build pruned student.

File quan trong:

```text
2_pruning/artifacts/blueprint.json
2_pruning/configs/pruning_config.json
2_pruning/metrics/pruning_summary.json
2_pruning/pruning_search_time.json
```

### Cac Strategy S1-S12

Code ho tro 12 strategy:

| Strategy | Method | Y nghia |
|---|---|---|
| `S1` | `static` | Giu top-k channel theo prune ratio co dinh |
| `S2` | `kneedle` | Tim threshold bang Kneedle |
| `S3` | `otsu` | Tim threshold bang Otsu |
| `S4` | `gmm` | Tim threshold bang Gaussian Mixture Model |
| `S5` | `middle_static` | Prune `conv2` trong ResNet bottleneck bang static ratio |
| `S6` | `middle_kneedle` | Prune `conv2` trong bottleneck bang Kneedle |
| `S7` | `middle_otsu` | Prune `conv2` trong bottleneck bang Otsu |
| `S8` | `middle_gmm` | Prune `conv2` trong bottleneck bang GMM |
| `S9` | `full_static` | Prune full bottleneck path, gom ca `conv3 output`, bang static ratio |
| `S10` | `full_kneedle` | Prune full bottleneck path bang Kneedle |
| `S11` | `full_otsu` | Prune full bottleneck path bang Otsu |
| `S12` | `full_gmm` | Prune full bottleneck path bang GMM |

Voi `S1-S4`, student la `PDGUNet`.

Voi `S5-S8`, student la `middle_pruned_resnet_unet`. Nhom nay chi prune channel giua `conv2` trong ResNet bottleneck, con `conv1`, output cua `conv3`, downsample va residual boundary duoc giu full de tranh vo shape.

Voi `S9-S12`, student la `full_pruning_resnet_unet`. Nhom nay prune ca main path cua bottleneck: `conv1`, `conv2`, `conv3 input` va `conv3 output/bn3`. Khi output channel cua block bi cat, code rebuild residual projection va decoder skip inputs de residual add/skip connection khong bi mismatch shape.

## 7. Static Ratio Va Threshold Search

`search_time_seconds` duoc do quanh ham:

```python
extract_pruned_blueprint(...)
```

No khong chi do rieng buoc tinh threshold, ma gom ca:

- Tim layer prune.
- Lay importance score.
- Chon channel giu/cat.
- Sort/rank channel.
- Tao metadata chi tiet cho `blueprint.json`.

Voi `static`, code van phai sort importance de lay top-k channel:

```python
np.argsort(scores)[::-1][:k]
```

Nen static ratio khong phai la khong co search. No chi co dinh so channel giu lai, nhung van phai tim channel nao co importance cao nhat.

## 8. Phase 3: Student Training

Phase `3_student` load student tu `2_pruning`, sau do tune/train lai student.

Loss tong:

```text
Ltotal = Lseg
       + lambda_distill * Ldistill
```

Hai che do chinh:

```text
use_kd_output=0  -> segmentation only
use_kd_output=1  -> segmentation + KD/distillation
```

Mac dinh quan trong:

- `Lseg`: segmentation loss, thuong la hybrid CE + Dice.
- `Ldistill`: distillation output logits tu teacher sang student.
- `use_sparsity`, `feature_distill`, va `aux_loss` chi con la flag tuong thich/ablation phu, mac dinh tat.

## 9. Gate/Sparsity Status

Gate/gated UNet khong con la pipeline chinh. `PDGUNet` hien la pruned UNet binh thuong, cac method gate chi con no-op de checkpoint/script cu khong vo.

Effective sparsity luon tat trong pipeline moi, nen output tag mac dinh se la `loss_seg_only` hoac `loss_seg_kd`, khong tu gan `_sparsity` khi khong co sparsity thuc su.

## 10. Checkpoint Va Log

Moi phase luu checkpoint trong:

```text
checkpoints/
```

Cac file thuong gap:

```text
best.pth
last.pth
train_log.csv
metrics.json
metadata/best.json
metadata/last.json
```

Voi student, `train_log.csv` ghi cac thong tin theo epoch nhu:

```text
val_macro_dice
val_total_loss
val_segmentation_loss
val_distillation_loss
val_sparsity_loss
val_feature_distill_loss
val_auxiliary_loss
```

Checkpoint `.pth` luu them:

- `config`
- `metrics`
- `model_info`
- `extra_state`
- `history`
- `epoch_diagnostics`

## 11. Evaluation

Sau khi train/prune, code evaluate tren cac split trong:

```text
final_eval_splits
```

Mac dinh thuong la:

```text
train val test
```

Metric chinh:

- Dice
- IoU
- HD95
- Params
- FLOPs
- FPS
- Inference latency

Output evaluation nam o:

```text
<phase>/evaluations/<split>/
<phase>/metrics/
<phase>/reports/
```

## 12. Visualization

Code xuat anh visualization cho prediction, ground truth va image goc.

Thu muc thuong gap:

```text
3_student/artifacts/visualizations/
3_student/evaluations/train/panel/
3_student/evaluations/val/panel/
3_student/evaluations/test/panel/
```

Visualization gom cac cot image, GT va prediction va da co text label tren panel:

```text
Image | GT | PR
```

## 13. Timing

Code do timing bang `time.perf_counter()`.

Y nghia cac chi so:

| Field | Y nghia |
|---|---|
| `search_time_seconds` | Thoi gian tao pruning blueprint va tim threshold/top-k |
| `pruning_time_seconds` | Thoi gian phase pruning truoc final evaluation |
| `training_time_seconds` | Thoi gian train teacher/student |
| `inference_time_seconds` | Latency trung binh cho 1 forward pass dummy input |
| `evaluation_time_seconds` | Tong thoi gian final evaluation tren cac split |
| `total_time_seconds` | Tong wall-clock time cua phase |

Voi phase `student`, `pruning_time_seconds` thuong la `0.0` vi pruning da nam o phase `2_pruning`.

Thoi gian tim threshold duoc luu o:

```text
2_pruning/pruning_search_time.json
2_pruning/metrics/pruning_search_time.json
2_pruning/artifacts/pruning_search_time.json
2_pruning/artifacts/blueprint.json
```

Gia tri threshold tung layer nam trong `blueprint.json`, key:

```text
pruning_threshold
```

## 14. Cach Chay Thuong Dung

Chay mot strategy Otsu khong step-3 pruning:

```bash
PRUNE_STRATEGY=S3 STEP3_PRUNING=0 bash run_pgd_kvasir_seg.sh
```

Chay Otsu co step-3 pruning:

```bash
PRUNE_STRATEGY=S3 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 bash run_pgd_kvasir_seg.sh
```

Chay static ratio 0.5:

```bash
PRUNE_STRATEGY=S1 PRUNE_RATE=0.5 bash run_pgd_kvasir_seg.sh
```

## 15. Ket Luan

Toan bo code dang thuc hien pipeline:

```text
Teacher lon
-> phan tich importance channel
-> prune bang S1-S12
-> build student nho hon
-> reuse weight tu teacher
-> distill/tune student
-> evaluate Dice/IoU/HD95, params, FLOPs, FPS, timing
-> export checkpoint, CSV, JSON, PDF va visualization
```

Muc tieu cuoi la chung minh pruned/distilled student co the dat performance gan bang hoac tot hon teacher/baseline, trong khi giam chi phi tinh toan.
