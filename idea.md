# Ý Tưởng Hiện Tại Của PGD-UNet

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

Tài liệu này phản ánh logic code hiện tại của project PGD-UNet. Các nội dung cũ về gate/gated UNet, sparsity là pipeline chính, hoặc S1-S8-only đã được thay bằng mô tả mới bên dưới.

## 1. Mục Tiêu Chính

Pipeline hiện tại tập trung vào **structural channel pruning + fine-tuning/KD** cho bài toán medical image segmentation.

Ý tưởng tổng quát:

```text
Teacher model
  -> phân tích importance của channel
  -> chọn channel giữ/prune bằng S1-S12
  -> sinh pruning blueprint
  -> build student đã prune
  -> copy subset weight từ teacher sang student
  -> fine-tune student
       A. segmentation only
       B. segmentation + KD/distillation
  -> evaluate Dice/IoU/HD95, Params, FLOPs, FPS, timing
```

Mục tiêu chính không phải bắt buộc Dice phải tăng, mà là tạo trade-off tốt hơn giữa chất lượng segmentation và chi phí tính toán.

## 2. UNetResNet152 Đang Là Gì?

File chính:

```text
networks/Basic_Model/Unet_restnet.py
```

Tên `UNetResNet152` nghĩa là:

```text
U-Net architecture with ResNet152 encoder
```

Không có nghĩa là cả encoder và decoder đều là ResNet.

## 3. Encoder Là ResNet152

Code lấy backbone từ:

```python
torchvision.models.resnet152(...)
```

Sau đó dùng các phần sau làm encoder:

```text
stem = conv1 + bn1 + relu
maxpool
layer1
layer2
layer3
layer4
```

Luồng feature:

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

Mỗi bottleneck block có dạng:

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

## 4. Decoder Vì Sao Là U-Net Style, Không Phải ResNet Style?

Trong code, decoder được định nghĩa bằng `DecoderBlock` và `FinalUpBlock`, không dùng ResNet bottleneck.

Decoder hiện tại:

```text
center   = DoubleConv2d(2048 -> 2048)
dec4     = ConvTranspose2d + concat skip layer3 + DoubleConv2d
dec3     = ConvTranspose2d + concat skip layer2 + DoubleConv2d
dec2     = ConvTranspose2d + concat skip layer1 + DoubleConv2d
dec1     = ConvTranspose2d + concat skip stem   + DoubleConv2d
final_up = Upsample + DoubleConv2d
head     = Conv2d 1x1
```

Lý do:

- ResNet152 gốc là classification backbone, chuyên downsample và extract feature, không có decoder upsampling.
- Segmentation cần khôi phục spatial resolution để tạo mask, nên cần decoder có upsample.
- U-Net decoder phù hợp vì nó fuse skip connection từ encoder ở nhiều scale.
- Nếu muốn decoder kiểu ResNet, phải tự thiết kế residual upsampling decoder riêng; code hiện tại không làm hướng đó.

Câu trình bày ngắn:

> Mô hình dùng ResNet152 làm encoder để tận dụng feature extractor mạnh, nhưng decoder vẫn là U-Net style vì segmentation cần upsample và fuse skip features. ResNet gốc không có decoder.

## 5. Channel Importance

Trước khi prune, mỗi channel được gán importance score.

Ưu tiên:

```text
importance(channel) = abs(BatchNorm weight)
```

Nếu không tìm được BatchNorm tương ứng, fallback:

```text
importance(channel) = L1 norm của convolution filter
```

Diễn giải:

```text
score cao  -> channel quan trọng hơn -> giữ
score thấp -> channel yếu hơn        -> prune
```

## 6. Mapping Strategy S1-S12

Mapping trong `train_pgd.py` hiện tại:

| Strategy | Method | Display | Ý tưởng |
|---|---|---|---|
| S1 | `static` | Static Blueprint | Stage/channel blueprint bằng static ratio |
| S2 | `kneedle` | Kneedle Blueprint | Stage/channel blueprint bằng Kneedle threshold |
| S3 | `otsu` | Otsu Blueprint | Stage/channel blueprint bằng Otsu threshold |
| S4 | `gmm` | GMM Blueprint | Stage/channel blueprint bằng GMM threshold |
| S5 | `middle_static` | Middle-Static Conv2 | ResNet bottleneck, chỉ prune `conv2` bằng static ratio |
| S6 | `middle_kneedle` | Middle-Kneedle Conv2 | ResNet bottleneck, chỉ prune `conv2` bằng Kneedle |
| S7 | `middle_otsu` | Middle-Otsu Conv2 | ResNet bottleneck, chỉ prune `conv2` bằng Otsu |
| S8 | `middle_gmm` | Middle-GMM Conv2 | ResNet bottleneck, chỉ prune `conv2` bằng GMM |
| S9 | `full_static` | Full-Static Block | Prune full bottleneck path, gồm cả `conv3 output`, bằng static ratio |
| S10 | `full_kneedle` | Full-Kneedle Block | Prune full bottleneck path bằng Kneedle |
| S11 | `full_otsu` | Full-Otsu Block | Prune full bottleneck path bằng Otsu |
| S12 | `full_gmm` | Full-GMM Block | Prune full bottleneck path bằng GMM |

## 7. Nhóm S1-S4: Blueprint Student

S1-S4 tạo student nhỏ theo stage/channel config.

Đặc điểm:

- Dùng các module/stage chính để tính importance.
- Sinh `channel_config` cho student.
- Khi `teacher_model=unet_plus_plus`, build `blueprint_unet_plus_plus`: mini UNet++ theo `channel_config`, decoder nested UNet++, không còn learnable gate.
- Khi teacher không phải UNet++, fallback về `PDGUNet` plain pruned UNet để giữ backward compatibility.
- Mapping layer/block không còn 1-1 với ResNet teacher, nên không nên ép vào layerwise bottleneck plot.

Phù hợp để visualize:

```text
stem/down1/down2/down3/down4 before-after channels
global prune ratio
Params/FLOPs/FPS/Dice tradeoff
```

## 8. Nhóm S5-S8: Middle Conv2 Pruning

S5-S8 dành cho `teacher_model=unet_plus_plus`.

Ý tưởng: giữ boundary của bottleneck an toàn, chỉ prune phần giữa `conv2`.

Pruned:

```text
conv2 output
bn2
conv3 input tương ứng
```

Protected:

```text
conv1/bn1
conv3 output/bn3
downsample
residual add
decoder/head
```

Vì sao chọn `conv2`?

- `conv1` và `conv3 output` liên quan trực tiếp đến input/output boundary của block.
- Nếu prune `conv3 output`, residual path và block sau cũng phải đổi shape.
- S5-S8 là hướng an toàn: giảm compute bên trong block nhưng giữ output shape.

Student architecture:

```text
middle_pruned_unet_plus_plus
```

Blueprint chính:

```text
middle_prune_plan
stage_middle_channel_config
teacher_vs_student_rows
global_pruning_summary
```

## 9. Nhóm S9-S12: Full Bottleneck Pruning

S9-S12 hiện đã được sửa thành full block pruning thật hơn: prune cả output của bottleneck, không chỉ internal width.

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

Khi `conv3 output` bị prune, code phải đồng bộ các phần phụ thuộc:

```text
input của block kế tiếp
downsample/residual projection
residual add
stage output channels
decoder skip input channels
UNet++ encoder feature widths
```

Trong `full_pruning_unet_plus_plus.py`, nếu student là UNet++ thì không copy các module custom `center/dec*` của `UNetResNet152`. Code prune bottleneck output rồi gắn stage expander 1x1 để restore feature width cho decoder SMP UNet++:

```text
decoder input = projected output cua layer4
skip layer3  = projected output cua layer3
skip layer2  = projected output cua layer2
skip layer1  = projected output cua layer1
stem skip    = stem 64 channels
```

Blueprint S9-S12 lưu thêm:

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

Câu trình bày:

> S9-S12 prune toàn bộ bottleneck path, bao gồm cả `conv3 output`. Vì output block đổi channel, pipeline phải rebuild residual projection và decoder skip inputs để residual add và U-Net skip connection không bị mismatch.

## 10. Các Thuật Toán Chọn Channel

### Static

Dùng prune ratio cố định:

```text
num_keep = ceil((1 - r) * C)
```

Áp dụng cho:

```text
S1, S5, S9
```

### Kneedle

Không cần prune ratio. Sort importance, chuẩn hóa curve, tìm điểm gãy, dùng điểm đó làm threshold.

Áp dụng cho:

```text
S2, S6, S10
```

### Otsu

Tìm threshold chia importance thành 2 nhóm thấp/cao, tương tự threshold ảnh.

Áp dụng cho:

```text
S3, S7, S11
```

### GMM

Fit Gaussian Mixture Model 2 components cho importance score, xem score thuộc nhóm yếu/mạnh.

Áp dụng cho:

```text
S4, S8, S12
```

## 11. Loss Hiện Tại

File chính:

```text
utils/compression_loss.py
```

Pipeline chính chỉ cần 2 mode:

```text
A. segmentation only
B. segmentation + KD/distillation
```

Segmentation loss hỗ trợ:

```text
ce
dice
hybrid = 0.5 CE + 0.5 Dice
```

KD/distillation loss hỗ trợ:

```text
mse
ce
dice
kl
hybrid
```

Tổng loss chính:

```text
Ltotal = Lseg
```

hoặc:

```text
Ltotal = Lseg + lambda_distill * Lkd
```

Các loss feature/aux/sparsity vẫn còn trong code để tương thích hoặc ablation, nhưng default pipeline không dùng:

```text
use_feature_distill = 0
use_aux_loss = 0
use_sparsity = 0
lambda_sparsity = 0
```

## 12. Gate Và Sparsity Đang Ở Trạng Thái Nào?

Gate không còn là pipeline chính.

`networks/PGD_Unet/gated_unet.py` hiện giữ các method tương thích:

```text
get_gate_tensors()
get_gate_modules()
set_gate_trainable()
force_gates_open()
get_gate_statistics()
```

Nhưng chúng là no-op hoặc trả về rỗng. Vì vậy:

```text
val_sparsity_loss = 0.0
```

là đúng trong pipeline hiện tại.

`_effective_sparsity_loss_enabled()` trong `train_pgd.py` hiện trả về `False`, nên output mới không tự thêm `_sparsity` vào `loss_tag`.

## 13. Output Folder Và Loss Tag

Output proposal hiện có dạng:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/<loss_tag>/<output_dir>/
```

Ví dụ:

```text
outputs/pgd_unet/kvasir_seg/unet_plus_plus_teacher/loss_seg_kd/output_s3_otsu_auto_no/
```

Loss tag chính:

```text
loss_seg_only
loss_seg_kd
```

Static strategy có ratio trong folder:

```text
output_s1_static_0.5_no
output_s5_middle_static_0.5_no
output_s9_full_static_0.5_no
```

Dynamic strategy dùng `auto`:

```text
output_s3_otsu_auto_no
output_s7_middle_otsu_auto_no
output_s11_full_otsu_auto_no
```

## 14. Train Log Và Log.txt

Mỗi phase có `log.txt`:

```text
1_teacher/log.txt
2_pruning/log.txt
3_student/log.txt
pipeline/log.txt
```

`pipeline/run.log` vẫn giữ để tương thích output cũ.

`checkpoints/train_log.csv` lưu metric theo epoch. Các cột quan trọng:

```text
epoch
train_total_loss
train_seg_loss
train_kd_loss
val_macro_dice hoặc val_dice
val_iou
val_hd95
learning_rate
best_val_dice
is_best
use_kd_output
loss_tag
```

Figure 17 đọc từ file này.

## 15. Visualization

Visualization panel hiện có label rõ:

```text
Image | GT | PR
```

Áp dụng cho:

```text
artifacts/visualizations/
evaluations/<split>/panel/
```

Không đổi tên file output nếu không cần.

## 16. Figures Và Statistics

Các figure liên quan:

```text
figure5_layerwise_pruning_ratio.pdf
figure5_s1_s4_blueprint_stage_pruning_ratio.pdf
figure5_s5_s8_middle_conv2_layerwise_pruning_ratio.pdf
figure5_s9_s12_full_block_layerwise_pruning_ratio.pdf
figure6_accuracy_efficiency_tradeoff.pdf
figure17_kd_vs_seg_validation_dice.pdf
```

Ý nghĩa:

- Figure 5 chỉ nên dùng cho S5-S12 vì nhóm này còn mapping block/layer với teacher.
- S1-S4 nên dùng stage-level channel config hoặc Params/FPS/Dice tradeoff.
- Figure 6 là trade-off **sau fine-tune**, đọc phase `3_student`.
- Figure 17 so sánh validation Dice giữa segmentation only và segmentation + KD.

Code hien tai xuat them 3 file figure5 rieng:

- S1-S4: stage-level blueprint index (`stem`, `down1`, `down2`, `down3`, `down4`).
- S5-S8: teacher ResNet bottleneck block index, prune `conv2`.
- S9-S12: teacher ResNet bottleneck block index, prune internal width va `conv3 output`.

## 17. Timing

Các field timing thường gặp:

```text
search_time_seconds
pruning_time_seconds
training_time_seconds
inference_time_seconds
evaluation_time_seconds
total_time_seconds
```

Ý nghĩa:

- `search_time_seconds`: thời gian tạo blueprint, tính importance, chọn threshold/top-k, tạo metadata.
- `pruning_time_seconds`: thời gian phase pruning trước evaluation.
- `training_time_seconds`: thời gian train/fine-tune.
- `inference_time_seconds`: latency forward pass.
- `evaluation_time_seconds`: thời gian evaluate các split.
- `total_time_seconds`: wall-clock time của phase.

Threshold từng layer/block nằm trong:

```text
2_pruning/artifacts/blueprint.json
2_pruning/pruning_search_time.json
2_pruning/metrics/pruning_search_time.json
2_pruning/artifacts/weight_transfer/weight_transfer.json
2_pruning/artifacts/weight_transfer/stage_transfer_rows.csv
2_pruning/artifacts/weight_transfer/block_transfer_rows.csv
2_pruning/artifacts/weight_transfer/decoder_subset_transfer_rows.csv
```

## 18. Câu Chốt Để Trình Bày

Cập nhật hiện tại: với `teacher_model=unet_plus_plus`, teacher/student chính dùng UNet++ với ResNet152 encoder. S1-S4 tính importance trên encoder để tạo mini `blueprint_unet_plus_plus`; S5-S8 copy UNet++ teacher rồi prune an toàn `conv2` trong từng bottleneck encoder; S9-S12 copy UNet++ teacher rồi prune full bottleneck encoder, gồm cả `conv3 output`, sau đó dùng stage expander để giữ feature width hợp với decoder SMP UNet++. Sau pruning, student được fine-tune bằng segmentation loss hoặc segmentation + KD. Gate/sparsity không còn là pipeline chính.

> PGD-UNet hiện là pipeline nén mô hình segmentation bằng structural channel pruning. Teacher thường là UNetResNet152, tức ResNet152 encoder + U-Net decoder. S1-S4 tạo student blueprint nhỏ theo stage; S5-S8 prune an toàn `conv2` trong bottleneck; S9-S12 prune full bottleneck path gồm cả `conv3 output` và rebuild residual/decoder shape. Sau pruning, student được fine-tune bằng segmentation loss hoặc segmentation + KD. Gate/sparsity không còn là pipeline chính.

## Basic Model Baselines Mới

Basic branch hiện được route qua:

```text
networks/net_factory.py
```

Danh sách model đăng ký:

- `unet`: U-Net 2D cơ bản.
- `resunet`: Residual U-Net.
- `vnet`: V-Net 2D.
- `unetr`: UNETR 2D.
- `unet_resnet152`: U-Net decoder với ResNet152 encoder.
- `att_unet`: Attention U-Net. Code mới nằm ở `networks/Basic_Model/attention_unet.py`, dựa theo idea reference trong `networks/Basic_Model/src_ref/Image_Segmentation/network.py`.
- `r2unet`: R2U-Net/Recurrent Residual U-Net. Code mới cũng nằm ở `networks/Basic_Model/attention_unet.py`, dùng recurrent residual blocks theo reference.
- `unet_plus_plus`: UNet++ wrapper nằm ở `networks/Basic_Model/unet_plus_plus.py`, dùng `segmentation_models_pytorch.UnetPlusPlus`.

`att_unet`, `r2unet`, và `unet_plus_plus` đều trả output theo chuẩn `BaseSegmentationModel`, nên `train_basic_model.py`, `test2d.py`, metrics, reports và visualization có thể dùng chung như các baseline cũ.
