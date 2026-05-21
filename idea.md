# Ã TÆ°á»Ÿng Hiá»‡n Táº¡i Cá»§a PGD-UNet

## Cap Nhat Kien Truc S1-S12

Pipeline S1-S12 mac dinh hien dung `unet_plus_plus` lam teacher chinh. `unet_plus_plus` trong code la `segmentation_models_pytorch.UnetPlusPlus` voi encoder `resnet152`. Do do importance/pruning bay gio doc cac stage encoder:

```text
model.encoder.conv1
model.encoder.layer1
model.encoder.layer2
model.encoder.layer3
model.encoder.layer4
```

S5-S8 va S9-S12 van prune ResNet bottleneck trong encoder, nhung student khi teacher la `unet_plus_plus` se giu decoder UNet++ thay vi quay ve decoder custom cua `UNetResNet152`.

TÃ i liá»‡u nÃ y pháº£n Ã¡nh logic code hiá»‡n táº¡i cá»§a project PGD-UNet. CÃ¡c ná»™i dung cÅ© vá» gate/gated UNet, sparsity lÃ  pipeline chÃ­nh, hoáº·c S1-S8-only Ä‘Ã£ Ä‘Æ°á»£c thay báº±ng mÃ´ táº£ má»›i bÃªn dÆ°á»›i.

## 1. Má»¥c TiÃªu ChÃ­nh

Pipeline hiá»‡n táº¡i táº­p trung vÃ o **structural channel pruning + fine-tuning/KD** cho bÃ i toÃ¡n medical image segmentation.

Ã tÆ°á»Ÿng tá»•ng quÃ¡t:

```text
Teacher model
  -> phÃ¢n tÃ­ch importance cá»§a channel
  -> chá»n channel giá»¯/prune báº±ng S1-S12
  -> sinh pruning blueprint
  -> build student Ä‘Ã£ prune
  -> copy subset weight tá»« teacher sang student
  -> fine-tune student
       A. segmentation only
       B. segmentation + KD/distillation
  -> evaluate Dice/IoU/HD95, Params, FLOPs, FPS, timing
```

Má»¥c tiÃªu chÃ­nh khÃ´ng pháº£i báº¯t buá»™c Dice pháº£i tÄƒng, mÃ  lÃ  táº¡o trade-off tá»‘t hÆ¡n giá»¯a cháº¥t lÆ°á»£ng segmentation vÃ  chi phÃ­ tÃ­nh toÃ¡n.

## 2. UNetResNet152 Äang LÃ  GÃ¬?

File chÃ­nh:

```text
networks/Basic_Model/Unet_restnet.py
```

TÃªn `UNetResNet152` nghÄ©a lÃ :

```text
U-Net architecture with ResNet152 encoder
```

KhÃ´ng cÃ³ nghÄ©a lÃ  cáº£ encoder vÃ  decoder Ä‘á»u lÃ  ResNet.

## 3. Encoder LÃ  ResNet152

Code láº¥y backbone tá»«:

```python
torchvision.models.resnet152(...)
```

Sau Ä‘Ã³ dÃ¹ng cÃ¡c pháº§n sau lÃ m encoder:

```text
stem = conv1 + bn1 + relu
maxpool
layer1
layer2
layer3
layer4
```

Luá»“ng feature:

```text
input
  -> stem      : 64 channels
  -> layer1    : 256 channels
  -> layer2    : 512 channels
  -> layer3    : 1024 channels
  -> layer4    : 2048 channels
```

Trong ResNet152:

```text
layer1 = 3 bottleneck blocks
layer2 = 8 bottleneck blocks
layer3 = 36 bottleneck blocks
layer4 = 3 bottleneck blocks
```

Má»—i bottleneck block cÃ³ dáº¡ng:

```text
input
  -> conv1 1x1
  -> bn1 + relu
  -> conv2 3x3
  -> bn2 + relu
  -> conv3 1x1
  -> bn3
  + residual/downsample
  -> relu
  -> output
```

## 4. Decoder VÃ¬ Sao LÃ  U-Net Style, KhÃ´ng Pháº£i ResNet Style?

Trong code, decoder Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a báº±ng `DecoderBlock` vÃ  `FinalUpBlock`, khÃ´ng dÃ¹ng ResNet bottleneck.

Decoder hiá»‡n táº¡i:

```text
center   = DoubleConv2d(2048 -> 2048)
dec4     = ConvTranspose2d + concat skip layer3 + DoubleConv2d
dec3     = ConvTranspose2d + concat skip layer2 + DoubleConv2d
dec2     = ConvTranspose2d + concat skip layer1 + DoubleConv2d
dec1     = ConvTranspose2d + concat skip stem   + DoubleConv2d
final_up = Upsample + DoubleConv2d
head     = Conv2d 1x1
```

LÃ½ do:

- ResNet152 gá»‘c lÃ  classification backbone, chuyÃªn downsample vÃ  extract feature, khÃ´ng cÃ³ decoder upsampling.
- Segmentation cáº§n khÃ´i phá»¥c spatial resolution Ä‘á»ƒ táº¡o mask, nÃªn cáº§n decoder cÃ³ upsample.
- U-Net decoder phÃ¹ há»£p vÃ¬ nÃ³ fuse skip connection tá»« encoder á»Ÿ nhiá»u scale.
- Náº¿u muá»‘n decoder kiá»ƒu ResNet, pháº£i tá»± thiáº¿t káº¿ residual upsampling decoder riÃªng; code hiá»‡n táº¡i khÃ´ng lÃ m hÆ°á»›ng Ä‘Ã³.

CÃ¢u trÃ¬nh bÃ y ngáº¯n:

> MÃ´ hÃ¬nh dÃ¹ng ResNet152 lÃ m encoder Ä‘á»ƒ táº­n dá»¥ng feature extractor máº¡nh, nhÆ°ng decoder váº«n lÃ  U-Net style vÃ¬ segmentation cáº§n upsample vÃ  fuse skip features. ResNet gá»‘c khÃ´ng cÃ³ decoder.

## 5. Channel Importance

TrÆ°á»›c khi prune, má»—i channel Ä‘Æ°á»£c gÃ¡n importance score.

Æ¯u tiÃªn:

```text
importance(channel) = abs(BatchNorm weight)
```

Náº¿u khÃ´ng tÃ¬m Ä‘Æ°á»£c BatchNorm tÆ°Æ¡ng á»©ng, fallback:

```text
importance(channel) = L1 norm cá»§a convolution filter
```

Diá»…n giáº£i:

```text
score cao  -> channel quan trá»ng hÆ¡n -> giá»¯
score tháº¥p -> channel yáº¿u hÆ¡n        -> prune
```

## 6. Mapping Strategy S1-S12

Mapping trong `train_pgd.py` hiá»‡n táº¡i:

| Strategy | Method | Display | Ã tÆ°á»Ÿng |
|---|---|---|---|
| S1 | `static` | Static Blueprint | Stage/channel blueprint báº±ng static ratio |
| S2 | `kneedle` | Kneedle Blueprint | Stage/channel blueprint báº±ng Kneedle threshold |
| S3 | `otsu` | Otsu Blueprint | Stage/channel blueprint báº±ng Otsu threshold |
| S4 | `gmm` | GMM Blueprint | Stage/channel blueprint báº±ng GMM threshold |
| S5 | `middle_static` | Middle-Static Conv2 | ResNet bottleneck, chá»‰ prune `conv2` báº±ng static ratio |
| S6 | `middle_kneedle` | Middle-Kneedle Conv2 | ResNet bottleneck, chá»‰ prune `conv2` báº±ng Kneedle |
| S7 | `middle_otsu` | Middle-Otsu Conv2 | ResNet bottleneck, chá»‰ prune `conv2` báº±ng Otsu |
| S8 | `middle_gmm` | Middle-GMM Conv2 | ResNet bottleneck, chá»‰ prune `conv2` báº±ng GMM |
| S9 | `full_static` | Full-Static Block | Prune full bottleneck path, gá»“m cáº£ `conv3 output`, báº±ng static ratio |
| S10 | `full_kneedle` | Full-Kneedle Block | Prune full bottleneck path báº±ng Kneedle |
| S11 | `full_otsu` | Full-Otsu Block | Prune full bottleneck path báº±ng Otsu |
| S12 | `full_gmm` | Full-GMM Block | Prune full bottleneck path báº±ng GMM |

## 7. NhÃ³m S1-S4: Blueprint Student

S1-S4 táº¡o student nhá» theo stage/channel config.

Äáº·c Ä‘iá»ƒm:

- DÃ¹ng cÃ¡c module/stage chÃ­nh Ä‘á»ƒ tÃ­nh importance.
- Sinh `channel_config` cho student.
- Khi `teacher_model=unet_plus_plus`, build `blueprint_unet_plus_plus`: mini UNet++ theo `channel_config`, decoder nested UNet++, khÃ´ng cÃ²n learnable gate.
- Khi teacher khÃ´ng pháº£i UNet++, fallback vá» `PDGUNet` plain pruned UNet Ä‘á»ƒ giá»¯ backward compatibility.
- Mapping layer/block khÃ´ng cÃ²n 1-1 vá»›i ResNet teacher, nÃªn khÃ´ng nÃªn Ã©p vÃ o layerwise bottleneck plot.

PhÃ¹ há»£p Ä‘á»ƒ visualize:

```text
stem/down1/down2/down3/down4 before-after channels
global prune ratio
Params/FLOPs/FPS/Dice tradeoff
```

## 8. NhÃ³m S5-S8: Middle Conv2 Pruning

S5-S8 dÃ nh cho `teacher_model=unet_plus_plus`.

Ã tÆ°á»Ÿng: giá»¯ boundary cá»§a bottleneck an toÃ n, chá»‰ prune pháº§n giá»¯a `conv2`.

Pruned:

```text
conv2 output
bn2
conv3 input tÆ°Æ¡ng á»©ng
```

Protected:

```text
conv1/bn1
conv3 output/bn3
downsample
residual add
decoder/head
```

VÃ¬ sao chá»n `conv2`?

- `conv1` vÃ  `conv3 output` liÃªn quan trá»±c tiáº¿p Ä‘áº¿n input/output boundary cá»§a block.
- Náº¿u prune `conv3 output`, residual path vÃ  block sau cÅ©ng pháº£i Ä‘á»•i shape.
- S5-S8 lÃ  hÆ°á»›ng an toÃ n: giáº£m compute bÃªn trong block nhÆ°ng giá»¯ output shape.

Student architecture:

```text
middle_pruned_unet_plus_plus
```

Blueprint chÃ­nh:

```text
middle_prune_plan
stage_middle_channel_config
teacher_vs_student_rows
global_pruning_summary
```

## 9. NhÃ³m S9-S12: Full Bottleneck Pruning

S9-S12 hiá»‡n Ä‘Ã£ Ä‘Æ°á»£c sá»­a thÃ nh full block pruning tháº­t hÆ¡n: prune cáº£ output cá»§a bottleneck, khÃ´ng chá»‰ internal width.

Student architecture:

```text
full_pruning_unet_plus_plus
```

Pruned trong main path:

```text
conv1 input/output
bn1
conv2 input/output
bn2
conv3 input/output
bn3
```

Khi `conv3 output` bá»‹ prune, code pháº£i Ä‘á»“ng bá»™ cÃ¡c pháº§n phá»¥ thuá»™c:

```text
input cá»§a block káº¿ tiáº¿p
downsample/residual projection
residual add
stage output channels
decoder skip input channels
UNet++ encoder feature widths
```

Trong `full_pruning_unet_plus_plus.py`, náº¿u student lÃ  UNet++ thÃ¬ khÃ´ng copy cÃ¡c module custom `center/dec*` cá»§a `UNetResNet152`. Code prune bottleneck output rá»“i gáº¯n stage expander 1x1 Ä‘á»ƒ restore feature width cho decoder SMP UNet++:

```text
decoder input = projected output cua layer4
skip layer3  = projected output cua layer3
skip layer2  = projected output cua layer2
skip layer1  = projected output cua layer1
stem skip    = stem 64 channels
```

Blueprint S9-S12 lÆ°u thÃªm:

```text
full_prune_plan
stage_full_channel_config
stage_full_output_channel_config
stage_output_kept_indices
input_channel_indices
internal_kept_channel_indices
output_kept_channel_indices
teacher_vs_student_rows
global_pruning_summary
student_architecture = full_pruning_unet_plus_plus
```

CÃ¢u trÃ¬nh bÃ y:

> S9-S12 prune toÃ n bá»™ bottleneck path, bao gá»“m cáº£ `conv3 output`. VÃ¬ output block Ä‘á»•i channel, pipeline pháº£i rebuild residual projection vÃ  decoder skip inputs Ä‘á»ƒ residual add vÃ  U-Net skip connection khÃ´ng bá»‹ mismatch.

## 10. CÃ¡c Thuáº­t ToÃ¡n Chá»n Channel

### Static

DÃ¹ng prune ratio cá»‘ Ä‘á»‹nh:

```text
num_keep = ceil((1 - r) * C)
```

Ãp dá»¥ng cho:

```text
S1, S5, S9
```

### Kneedle

KhÃ´ng cáº§n prune ratio. Sort importance, chuáº©n hÃ³a curve, tÃ¬m Ä‘iá»ƒm gÃ£y, dÃ¹ng Ä‘iá»ƒm Ä‘Ã³ lÃ m threshold.

Ãp dá»¥ng cho:

```text
S2, S6, S10
```

### Otsu

TÃ¬m threshold chia importance thÃ nh 2 nhÃ³m tháº¥p/cao, tÆ°Æ¡ng tá»± threshold áº£nh.

Ãp dá»¥ng cho:

```text
S3, S7, S11
```

### GMM

Fit Gaussian Mixture Model 2 components cho importance score, xem score thuá»™c nhÃ³m yáº¿u/máº¡nh.

Ãp dá»¥ng cho:

```text
S4, S8, S12
```

## 11. Loss Hiá»‡n Táº¡i

File chÃ­nh:

```text
utils/compression_loss.py
```

Pipeline chÃ­nh chá»‰ cáº§n 2 mode:

```text
A. segmentation only
B. segmentation + KD/distillation
```

Segmentation loss há»— trá»£:

```text
ce
dice
hybrid = 0.5 CE + 0.5 Dice
```

KD/distillation loss há»— trá»£:

```text
mse
ce
dice
kl
hybrid
```

Tá»•ng loss chÃ­nh:

```text
Ltotal = Lseg
```

hoáº·c:

```text
Ltotal = Lseg + lambda_distill * Lkd
```

CÃ¡c loss feature/aux/sparsity váº«n cÃ²n trong code Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch hoáº·c ablation, nhÆ°ng default pipeline khÃ´ng dÃ¹ng:

```text
use_feature_distill = 0
use_aux_loss = 0
use_sparsity = 0
lambda_sparsity = 0
```

## 12. Gate VÃ  Sparsity Äang á»ž Tráº¡ng ThÃ¡i NÃ o?

Gate khÃ´ng cÃ²n lÃ  pipeline chÃ­nh.

`networks/PGD_Unet/gated_unet.py` hiá»‡n giá»¯ cÃ¡c method tÆ°Æ¡ng thÃ­ch:

```text
get_gate_tensors()
get_gate_modules()
set_gate_trainable()
force_gates_open()
get_gate_statistics()
```

NhÆ°ng chÃºng lÃ  no-op hoáº·c tráº£ vá» rá»—ng. VÃ¬ váº­y:

```text
val_sparsity_loss = 0.0
```

lÃ  Ä‘Ãºng trong pipeline hiá»‡n táº¡i.

`_effective_sparsity_loss_enabled()` trong `train_pgd.py` hiá»‡n tráº£ vá» `False`, nÃªn output má»›i khÃ´ng tá»± thÃªm `_sparsity` vÃ o `loss_tag`.

## 13. Output Folder VÃ  Loss Tag

Output proposal hiá»‡n cÃ³ dáº¡ng:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/<loss_tag>/<output_dir>/
```

VÃ­ dá»¥:

```text
outputs/pgd_unet/kvasir_seg/unet_plus_plus_teacher/loss_seg_kd/output_s3_otsu_auto_no/
```

Loss tag chÃ­nh:

```text
loss_seg_only
loss_seg_kd
```

Static strategy cÃ³ ratio trong folder:

```text
output_s1_static_0.5_no
output_s5_middle_static_0.5_no
output_s9_full_static_0.5_no
```

Dynamic strategy dÃ¹ng `auto`:

```text
output_s3_otsu_auto_no
output_s7_middle_otsu_auto_no
output_s11_full_otsu_auto_no
```

## 14. Train Log VÃ  Log.txt

Má»—i phase cÃ³ `log.txt`:

```text
1_teacher/log.txt
2_pruning/log.txt
3_student/log.txt
pipeline/log.txt
```

`pipeline/run.log` váº«n giá»¯ Ä‘á»ƒ tÆ°Æ¡ng thÃ­ch output cÅ©.

`checkpoints/train_log.csv` lÆ°u metric theo epoch. CÃ¡c cá»™t quan trá»ng:

```text
epoch
train_total_loss
train_seg_loss
train_kd_loss
val_macro_dice hoáº·c val_dice
val_iou
val_hd95
learning_rate
best_val_dice
is_best
use_kd_output
loss_tag
```

Figure 17 Ä‘á»c tá»« file nÃ y.

## 15. Visualization

Visualization panel hiá»‡n cÃ³ label rÃµ:

```text
Image | GT | PR
```

Ãp dá»¥ng cho:

```text
artifacts/visualizations/
evaluations/<split>/panel/
```

KhÃ´ng Ä‘á»•i tÃªn file output náº¿u khÃ´ng cáº§n.

## 16. Figures VÃ  Statistics

CÃ¡c figure liÃªn quan:

```text
figure5_layerwise_pruning_ratio.pdf
figure5_s1_s4_blueprint_stage_pruning_ratio.pdf
figure5_s5_s8_middle_conv2_layerwise_pruning_ratio.pdf
figure5_s9_s12_full_block_layerwise_pruning_ratio.pdf
figure6_accuracy_efficiency_tradeoff.pdf
figure17_kd_vs_seg_validation_dice.pdf
```

Ã nghÄ©a:

- Figure 5 chá»‰ nÃªn dÃ¹ng cho S5-S12 vÃ¬ nhÃ³m nÃ y cÃ²n mapping block/layer vá»›i teacher.
- S1-S4 nÃªn dÃ¹ng stage-level channel config hoáº·c Params/FPS/Dice tradeoff.
- Figure 6 lÃ  trade-off **sau fine-tune**, Ä‘á»c phase `3_student`.
- Figure 17 so sÃ¡nh validation Dice giá»¯a segmentation only vÃ  segmentation + KD.

Code hien tai xuat them 3 file figure5 rieng:

- S1-S4: stage-level blueprint index (`stem`, `down1`, `down2`, `down3`, `down4`).
- S5-S8: teacher ResNet bottleneck block index, prune `conv2`.
- S9-S12: teacher ResNet bottleneck block index, prune internal width va `conv3 output`.

## 17. Timing

CÃ¡c field timing thÆ°á»ng gáº·p:

```text
search_time_seconds
pruning_time_seconds
training_time_seconds
inference_time_seconds
evaluation_time_seconds
total_time_seconds
```

Ã nghÄ©a:

- `search_time_seconds`: thá»i gian táº¡o blueprint, tÃ­nh importance, chá»n threshold/top-k, táº¡o metadata.
- `pruning_time_seconds`: thá»i gian phase pruning trÆ°á»›c evaluation.
- `training_time_seconds`: thá»i gian train/fine-tune.
- `inference_time_seconds`: latency forward pass.
- `evaluation_time_seconds`: thá»i gian evaluate cÃ¡c split.
- `total_time_seconds`: wall-clock time cá»§a phase.

Threshold tá»«ng layer/block náº±m trong:

```text
2_pruning/artifacts/blueprint.json
2_pruning/pruning_search_time.json
2_pruning/metrics/pruning_search_time.json
2_pruning/artifacts/weight_transfer/weight_transfer.json
2_pruning/artifacts/weight_transfer/stage_transfer_rows.csv
2_pruning/artifacts/weight_transfer/block_transfer_rows.csv
2_pruning/artifacts/weight_transfer/decoder_subset_transfer_rows.csv
```

## 18. CÃ¢u Chá»‘t Äá»ƒ TrÃ¬nh BÃ y

Cap nhat hien tai: voi `teacher_model=unet_plus_plus`, teacher/student chinh dung UNet++ voi ResNet152 encoder. S1-S4 tinh importance tren encoder de tao mini `blueprint_unet_plus_plus`; S5-S8 copy UNet++ teacher roi prune an toan `conv2` trong tung bottleneck encoder; S9-S12 copy UNet++ teacher roi prune full bottleneck encoder, gom ca `conv3 output`, sau do dung stage expander de giu feature width hop voi decoder SMP UNet++. Sau pruning, student duoc fine-tune bang segmentation loss hoac segmentation + KD. Gate/sparsity khong con la pipeline chinh.

> PGD-UNet hiá»‡n lÃ  pipeline nÃ©n mÃ´ hÃ¬nh segmentation báº±ng structural channel pruning. Teacher thÆ°á»ng lÃ  UNetResNet152, tá»©c ResNet152 encoder + U-Net decoder. S1-S4 táº¡o student blueprint nhá» theo stage; S5-S8 prune an toÃ n `conv2` trong bottleneck; S9-S12 prune full bottleneck path gá»“m cáº£ `conv3 output` vÃ  rebuild residual/decoder shape. Sau pruning, student Ä‘Æ°á»£c fine-tune báº±ng segmentation loss hoáº·c segmentation + KD. Gate/sparsity khÃ´ng cÃ²n lÃ  pipeline chÃ­nh.
## Basic Model Baselines Moi

Basic branch hien duoc route qua:

```text
networks/net_factory.py
```

Danh sach model dang ky:

- `unet`: U-Net 2D co ban.
- `resunet`: Residual U-Net.
- `vnet`: V-Net 2D.
- `unetr`: UNETR 2D.
- `unet_resnet152`: U-Net decoder voi ResNet152 encoder.
- `att_unet`: Attention U-Net. Code moi nam o `networks/Basic_Model/attention_unet.py`, dua theo idea reference trong `networks/Basic_Model/src_ref/Image_Segmentation/network.py`.
- `r2unet`: R2U-Net/Recurrent Residual U-Net. Code moi cung nam o `networks/Basic_Model/attention_unet.py`, dung recurrent residual blocks theo reference.
- `unet_plus_plus`: UNet++ wrapper nam o `networks/Basic_Model/unet_plus_plus.py`, dung `segmentation_models_pytorch.UnetPlusPlus`.

`att_unet`, `r2unet`, va `unet_plus_plus` deu tra output theo chuan `BaseSegmentationModel`, nen `train_basic_model.py`, `test2d.py`, metrics, reports va visualization co the dung chung nhu cac baseline cu.
