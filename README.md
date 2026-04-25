# Medical Image Segmentation Codebase

Codebase PyTorch cho bài toán medical image segmentation 2D, được tổ chức theo 2 nhánh rõ ràng:

- `basic branch`: train và benchmark các baseline như `unet`, `unet_resnet152`, `resunet`, `vnet`, `unetr`
- `proposal branch`: pipeline cho `pdg_unet` gồm `teacher -> pruning -> student`, với output chuẩn nằm dưới `outputs/pgd_unet/...`

Mục tiêu của repo là chuẩn hóa:

- interface output của model
- output directory
- checkpoint / weight management
- metrics / reports / artifacts
- so sánh công bằng giữa baseline và proposal

## 1. Cấu trúc chính

```text
Code_main/
├─ dataloaders/
│  └─ dataset.py
├─ networks/
│  ├─ Basic_Model/
│  ├─ PGD_Unet/
│  │  ├─ gated_unet.py
│  │  ├─ middle_pruned_resnet_unet.py
│  │  ├─ pruning.py
│  │  ├─ pruning_algorithms/
│  │  │  └─ Kneedle_Otsu_GMM.py
│  │  └─ prunning.py
│  └─ net_factory.py
├─ utils/
│  ├─ channel_analysis.py
│  ├─ checkpoint_resolver.py
│  ├─ checkpoints.py
│  ├─ compression_loss.py
│  ├─ evaluation.py
│  ├─ experiment.py
│  ├─ losses.py
│  ├─ model_output.py
│  ├─ profiling.py
│  ├─ reporting.py
│  ├─ val_2d.py
│  └─ visualization.py
├─ train_basic_model.py
├─ train_pgd.py
├─ test2d.py
├─ compare_artifacts.py
└─ outputs/
```

Lưu ý:

- `networks/net_factory.py` chỉ route cho `basic branch`
- `networks/PGD_Unet/prunning.py` được giữ lại để tương thích import cũ

## 2. Chuẩn output model

Mọi model đều được chuẩn hóa qua `SegmentationModelOutput` trong `utils/model_output.py`.

Output chuẩn gồm:

- `logits`
- `probs`
- `preds`
- `features`
- `aux`
- `model_name`
- `backbone_name`
- `student_name`
- `phase_name`

Điều này giúp `train_basic_model.py`, `train_pgd.py`, `test2d.py`, phần evaluate và phần export artifact dùng lại được cùng logic.

## 3. Môi trường

Khuyến nghị:

- `Python 3.10+`
- `PyTorch` theo môi trường GPU bạn đang dùng

Cài dependency:

```bash
pip install -r requirements.txt
```

Nếu muốn tính FLOPs:

```bash
pip install thop
```

## 4. Dataset

Repo hiện hỗ trợ các dataset key:

- `cvc_clinicdb`
- `kvasir_seg`
- `etis`
- `etis_larib`
- `cvc_colondb`
- `cvc_colon_db`
- `cvc_300`
- `cvc300`
- `cvc` và `kvasir` vẫn còn là alias tương thích cũ, nhưng output mới nên dùng `cvc_clinicdb` và `kvasir_seg`
- `generic`

Ví dụ cấu trúc:

```text
data/
└─ Kvasir-SEG/
   ├─ images/
   ├─ masks/
   ├─ train.txt
   ├─ val.txt
   └─ test.txt
```

`dataloaders/dataset.py` hỗ trợ:

- quét cặp `image/mask`
- split file `train.txt`, `val.txt`, `test.txt`
- normalize mask nhị phân về `0/1`
- transform train / eval riêng

### Stable split manifests

Chuẩn hóa split cố định:

```bash
python analysis_data/generate_splits.py --dataset all --extract --seed 1337
```

Script này tạo manifest trong `data/<dataset>/splits/`. Mặc định với CVC-ClinicDB, ETIS, CVC-ColonDB và CVC-300: `train=int(total*0.8)`, `val=int(total*0.1)`, `test` lấy phần còn lại. Các dataset polyp hiện dùng chung split manifest để tránh mỗi lần chạy tự chia khác nhau.

Riêng `kvasir_seg` dùng policy theo manifest gốc: bắt buộc có `data/Kvasir-SEG/train.txt` và `data/Kvasir-SEG/val.txt`; `val.txt` gốc được dùng làm `splits/test.txt`, còn `train.txt` gốc được chia ổn định `90/10` thành `splits/train.txt` và `splits/val.txt`. Nếu thiếu file manifest gốc, script sẽ dừng lỗi thay vì tự quét và chia toàn bộ ảnh.

Nếu split đã tồn tại, script sẽ giữ nguyên và báo skip. Chỉ thêm `--overwrite` khi bạn cố ý muốn tạo lại split.

## 5. Basic branch

### Train

Ví dụ:

```bash
python train_basic_model.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

Ví dụ với `unet_resnet152`:

```bash
python train_basic_model.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

Lưu ý:

- với `unet_resnet152`, `encoder_pretrained` hiện mặc định là `1`
- nghĩa là nếu bạn không truyền cờ này, repo vẫn sẽ build encoder ResNet152 từ pretrained weights

### Evaluate

```bash
python test2d.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --split test
```

### Checkpoint behavior

`train_basic_model.py` sẽ:

- kiểm tra checkpoint compatible trong `outputs/<model>/<dataset>/checkpoints/`
- nếu có và không bật `--force_retrain 1` thì load lại, skip train, rồi vẫn export final evaluation / reports / artifacts
- nếu chưa có checkpoint phù hợp thì mới train

## 6. Proposal branch: PDG-UNet

### Train full pipeline

```bash
python run_pgd.py --dataset kvasir_seg --root_path data/Kvasir-SEG --teacher_model unet_resnet152 --exp pdg_kvasir_seg --max_epochs_teacher 50 --max_epochs_student 100 --prune_strategy S1 --prune_method static --static_prune_ratio 0.5 --lambda_distill 0.3 --lambda_sparsity 0.3 --batch_size 8 --patch_size 256 256
```

Lưu ý:

- với `teacher_model=unet_resnet152`, `--encoder_pretrained` hiện mặc định là `1`
- nên nếu teacher phải train lại từ đầu ở step 1, encoder ResNet152 sẽ mặc định dùng pretrained weights
- nếu bạn muốn tắt pretrained và train teacher từ random backbone, hãy truyền `--encoder_pretrained 0`

### Pruning strategy S1/S2/S3/S4/S5/S6/S7/S8

Pipeline hiện hỗ trợ 8 strategy ở step 2:

| Strategy | Method nội bộ | Ý nghĩa | Cần prune rate |
|---|---|---|---|
| `S1` | `static` | giữ top-k channel theo tỉ lệ cố định | có, qua `PRUNE_RATE` hoặc `--static_prune_ratio` |
| `S2` | `kneedle` | dynamic threshold bằng Kneedle | không |
| `S3` | `otsu` | dynamic threshold bằng Otsu | không |
| `S4` | `gmm` | dynamic threshold bằng Gaussian Mixture Model | không |
| `S5` | `middle_static` | trong ResNet152 bottleneck, giữ boundary layer full và chỉ prune `conv2` bằng static top-k như S1 | có, qua `PRUNE_RATE` hoặc `--static_prune_ratio` |
| `S6` | `middle_kneedle` | giống S5 về cấu trúc block, nhưng mask của `conv2` lấy bằng Kneedle riêng cho từng bottleneck như S2 | không |
| `S7` | `middle_otsu` | giống S5 về cấu trúc block, nhưng mask của `conv2` lấy bằng Otsu riêng cho từng bottleneck như S3 | không |
| `S8` | `middle_gmm` | giống S5 về cấu trúc block, nhưng mask của `conv2` lấy bằng GMM riêng cho từng bottleneck như S4 | không |

Với `S1/static` hoặc `S5/middle_static`, nếu prune rate là `r`, mỗi layer được prune sẽ giữ:

```text
ceil((1 - r) * num_channels)
```

channel có importance score cao nhất. `static_prune_ratio` phải nằm trong `[0, 1)` và luôn giữ ít nhất 1 channel.

Riêng `S5/middle_static`, `S6/middle_kneedle`, `S7/middle_otsu` và `S8/middle_gmm` hiện được thiết kế cho `teacher_model=unet_resnet152`. Với mỗi bottleneck block trong `layer1/layer2/layer3/layer4`, pipeline dùng `conv2` làm layer giữa để prune. `conv1`/`bn1`, output của `conv3`/`bn3`, downsample và shape residual được giữ full để đầu/cuối block không bị cắt. Vì `conv2` output bị prune thật, `conv3` sẽ chỉ copy input slice tương ứng các channel `conv2` được giữ, nhưng output channel của `conv3` vẫn giữ nguyên. Các strategy này dùng student kiến trúc `middle_pruned_resnet_unet`, nên PDG gate, sparsity loss và late hard pruning ở step 3 được tắt.

Interface khuyến nghị:

- shell ngoài cùng dùng `PRUNE_STRATEGY=S1/S2/S3/S4/S5/S6/S7/S8`
- Python nội bộ dùng `--prune_method static/kneedle/otsu/gmm/middle_static/middle_kneedle/middle_otsu/middle_gmm`
- `--prune_ratio` vẫn được giữ để backward compatibility, nhưng nên dùng `--static_prune_ratio` cho `static`

Ví dụ chạy trực tiếp Python:

```bash
# S1: static pruning 50%
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S1 --static_prune_ratio 0.5

# S2: Kneedle
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S2

# S3: Otsu
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S3

# S4: GMM
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S4

# S5: middle-static pruning 50%
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S5 --static_prune_ratio 0.5

# S6: middle-kneedle pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S6

# S7: middle-otsu pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S7

# S8: middle-gmm pruning
python run_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --prune_strategy S8
```

Khi chạy qua script theo dataset, set biến môi trường:

```bash
# S1: static, có pruning ở step 3 trong 4 epoch cuối
PRUNE_STRATEGY=S1 PRUNE_RATE=0.5 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S1: static, không pruning ở step 3
PRUNE_STRATEGY=S1 PRUNE_RATE=0.5 STEP3_PRUNING=0 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S2/S3/S4: dynamic, PRUNE_RATE không dùng
PRUNE_STRATEGY=S2 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S3 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S4 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S5: middle-static, dùng PRUNE_RATE giống S1
PRUNE_STRATEGY=S5 PRUNE_RATE=0.5 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S6: middle-kneedle, PRUNE_RATE không dùng
PRUNE_STRATEGY=S6 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh

# S7/S8: middle dynamic, PRUNE_RATE không dùng
PRUNE_STRATEGY=S7 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
PRUNE_STRATEGY=S8 STEP3_PRUNING=1 STEP3_PRUNING_EPOCHS=4 TEACHER_OUTPUT_ROOT=outputs bash run_pgd_cvc_clinicdb.sh
```

`run_pgd_common.sh` sẽ tự map strategy và tạo thư mục experiment con theo format:

```text
output_<prune_method>_<rate-or-auto>_<step3-pruning-epochs-or-no>
```

Script chạy theo dataset:

```bash
bash run_pgd_cvc_clinicdb.sh
bash run_pgd_kvasir_seg.sh
bash run_pgd_etis.sh
bash run_pgd_cvc_colondb.sh
bash run_pgd_cvc_300.sh
```

Đổi kiến trúc teacher bằng biến môi trường, ví dụ:

```bash
TEACHER_MODEL=unet_resnet152 PRUNE_STRATEGY=S1 bash run_pgd_etis.sh
```

| Strategy | Experiment folder |
|---|---|
| `S1`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_static_0.5_4` |
| `S1`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=0` | `output_static_0.5_no` |
| `S2`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_kneedle_auto_4` |
| `S3`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_otsu_auto_4` |
| `S4`, `STEP3_PRUNING=0` | `output_gmm_auto_no` |
| `S5`, `PRUNE_RATE=0.5`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_middle_static_0.5_4` |
| `S6`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_middle_kneedle_auto_4` |
| `S7`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_middle_otsu_auto_4` |
| `S8`, `STEP3_PRUNING=1`, `STEP3_PRUNING_EPOCHS=4` | `output_middle_gmm_auto_4` |

Log khi chạy sẽ có dạng:

```text
Pruning strategy: static
Static prune ratio: 0.5
Step-3 pruning: enabled
Step-3 pruning epochs: 4
Experiment folder: output_static_0.5_4
Teacher output root: outputs
Loss ablation: kd=1 sparsity=1 feat=0 aux=0
Loss weights: kd=0.3 sparsity=0.3 feat=0.1 aux=0.2
```

### Loss ablation cho Step 3

PDG student training có thể bật/tắt từng loss bằng argument hoặc biến môi trường trong các script dataset.

Các argument chính:

- `--use_kd_output 1/0`, mặc định `1`
- `--use_sparsity 1/0`, mặc định `1`
- `--use_feature_distill 1/0`, mặc định `0`
- `--use_aux_loss 1/0`, mặc định `0`
- `--lambda_feat`, mặc định `0.1`
- `--lambda_aux`, mặc định `0.2`
- `--feature_layers`, mặc định `bottleneck`
- mặc định feature distillation chỉ so tensor `bottleneck` ở điểm giao giữa encoder và decoder; loss này vẫn backprop qua feature của student nên encoder student được update weight

Tổng loss:

```text
Ltotal = Lseg
       + lambda_distill * L_KD_output      nếu use_kd_output=1
       + lambda_feat * L_feature_distill   nếu use_feature_distill=1
       + lambda_aux * L_aux                nếu use_aux_loss=1
       + lambda_sparsity * L_sparsity      nếu use_sparsity=1
```

Nếu loss nào tắt, history và metrics vẫn ghi loss đó bằng `0.0` để dễ vẽ biểu đồ.

`loss_tag` được tự sinh theo loss đang bật:

| Config | loss_tag |
|---|---|
| segmentation only | `loss_seg_only` |
| segmentation + KD | `loss_seg_kd` |
| segmentation + KD + sparsity | `loss_seg_kd_sparsity` |
| segmentation + KD + feature distill | `loss_seg_kd_feat` |
| segmentation + KD + feature distill + aux + sparsity | `loss_seg_kd_feat_aux_sparsity` |

Vì có `loss_tag`, output proposal hiện có dạng:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/<loss_tag>/<experiment_folder>/
```

Ví dụ:

```text
outputs/pgd_unet/cvc_clinicdb/unet_resnet152_teacher/loss_seg_kd_sparsity/output_gmm_auto_no/
```

Teacher vẫn dùng chung ở:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/
```

Nghĩa là đổi ablation loss không làm đổi logic pretrain/load/reuse teacher.

Ví dụ chạy ablation qua script dataset:

```bash
# Segmentation only
USE_KD_OUTPUT=0 USE_SPARSITY=0 USE_FEATURE_DISTILL=0 USE_AUX_LOSS=0 bash run_pgd_cvc_clinicdb.sh

# Segmentation + KD
USE_KD_OUTPUT=1 USE_SPARSITY=0 USE_FEATURE_DISTILL=0 USE_AUX_LOSS=0 bash run_pgd_cvc_clinicdb.sh

# Segmentation + KD + feature + aux + sparsity
USE_KD_OUTPUT=1 USE_SPARSITY=1 USE_FEATURE_DISTILL=1 USE_AUX_LOSS=1 LAMBDA_FEAT=0.1 LAMBDA_AUX=0.2 bash run_pgd_cvc_clinicdb.sh
```

`TEACHER_OUTPUT_ROOT` dùng để cố định nơi lưu `1_teacher`. Mặc định các script dataset dùng `outputs`, nên teacher chỉ cần train một lần tại:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/
```

Nếu trước đó teacher đã được train trong một output thử nghiệm cũ, pipeline sẽ cố gắng tìm checkpoint teacher compatible ở output hiện tại và register/copy nó sang `TEACHER_OUTPUT_ROOT` trước khi quyết định train teacher lại.

Các lần thử pruning khác nhau vẫn ghi vào folder con bên trong cùng root teacher, ví dụ:

```text
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_static_0.5_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_static_0.5_no/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_kneedle_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_middle_static_0.5_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_middle_kneedle_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_middle_otsu_auto_4/2_pruning/
outputs/pgd_unet/<dataset>/<teacher_model>_teacher/output_middle_gmm_auto_4/2_pruning/
```

### Teacher reuse

Khi cần teacher checkpoint, `train_pgd.py` sẽ check theo thứ tự:

1. explicit `--teacher_checkpoint` nếu có truyền vào
2. `outputs/pgd_unet/<dataset>/<teacher>_teacher/1_teacher/checkpoints/`
3. `outputs/<teacher_model>/<dataset>/checkpoints/`

Nếu reuse từ `basic branch`, checkpoint sẽ được register lại vào `1_teacher/checkpoints/` để output proposal luôn đầy đủ và dễ đọc.

Mặc định repo **không train lại teacher nếu đã có checkpoint compatible**. Chỉ khi cả 3 nguồn trên đều không có checkpoint phù hợp, hoặc bạn bật `--force_retrain_teacher 1`, thì `1_teacher` mới train lại từ đầu.

Nếu teacher phải train lại từ đầu và `teacher_model=unet_resnet152`, mặc định repo sẽ dùng `encoder_pretrained=1`.

### Cách tận dụng weight teacher để không cần train lại

Có 2 cách dùng thực tế:

#### Cách 1: train basic model trước, rồi để proposal tự reuse

Ví dụ train baseline teacher trước:

```bash
python train_basic_model.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

Sau đó chạy proposal với cùng `teacher_model` và `dataset`:

```bash
python train_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --encoder_pretrained 1 --max_epochs_teacher 50 --max_epochs_student 100 --prune_strategy S1 --static_prune_ratio 0.5
```

Khi đó `train_pgd.py` sẽ:

- tìm teacher checkpoint trong `outputs/unet_resnet152/cvc_clinicdb/checkpoints/`
- load lại weight đó
- register/copy checkpoint vào `outputs/pgd_unet/cvc_clinicdb/unet_resnet152_teacher/1_teacher/checkpoints/`
- skip teacher training nếu checkpoint compatible

#### Cách 2: chỉ định trực tiếp teacher checkpoint

Bạn cũng có thể chỉ định thẳng một checkpoint đã có:

```bash
python train_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --teacher_checkpoint outputs/unet_resnet152/cvc_clinicdb/checkpoints/best.pth --encoder_pretrained 1 --max_epochs_student 100 --prune_strategy S1 --static_prune_ratio 0.5
```

Repo sẽ kiểm tra compatibility của checkpoint này trước khi dùng. Nếu checkpoint này tồn tại và compatible, nó sẽ được ưu tiên dùng trước proposal outputs và basic outputs. Nếu checkpoint không tồn tại hoặc không tương thích, pipeline mới tiếp tục check các nguồn còn lại.

Khi `--teacher_checkpoint` hợp lệ, pipeline sẽ:

- load trực tiếp weight từ đúng path bạn truyền vào
- sau đó register/mirror checkpoint đó vào `outputs/pgd_unet/.../1_teacher/checkpoints/`
- dùng bản register này như proposal-owned checkpoint để export artifact và dễ tái sử dụng về sau

#### Khi nào proposal sẽ train lại teacher?

Teacher chỉ train lại khi:

- không có checkpoint compatible trong proposal outputs
- không có checkpoint compatible từ `--teacher_checkpoint`
- không có checkpoint compatible trong basic outputs
- hoặc bạn chủ động bật `--force_retrain_teacher 1`

### Pipeline phases

Pipeline proposal gồm:

1. `1_teacher`
2. `2_pruning`
3. `3_student`
4. `student_final`
5. `pipeline`

### Step 2: Pruning / pruned-student initialization

Step 2 hiện không chỉ sinh `blueprint` rồi dựng một student mới hoàn toàn.

Luồng hiện tại là:

1. phân tích teacher và chọn `kept_channel_indices` theo `prune_method`
2. sinh `blueprint` và `channel_config` cho `PDGUNet`
3. build `pruned student` từ blueprint
4. nạp lại weight của teacher ứng với các channel được giữ lại vào từng stage encoder của student
5. copy thêm các tensor còn tương thích bằng exact-match fallback
6. force gate mở ra để baseline ở `2_pruning` không bị suy giảm do gate mặc định
7. evaluate `train / val / test`

Điều này có nghĩa:

- `2_pruning` không còn là `random init baseline`
- student ở bước này đã reuse lại phần weight sống sót từ teacher whenever compatible
- metric ở `2_pruning` vì vậy phản ánh `pruned student after structural pruning + teacher weight subset reuse`, chứ không phải một model mới hoàn toàn từ đầu
- `blueprint.json` lưu thêm `prune_strategy`, `prune_method`, `static_prune_ratio`, `pruning_threshold` và `kept_channel_indices` để trace lại strategy đã dùng

### Step 2 weight-transfer logging

Khi chạy `2_pruning`, repo sẽ log rõ cho từng stage:

- thành phần nào `copy trực tiếp`
- thành phần nào `resize kernel rồi copy`
- thành phần nào `không map được`

Ví dụ log:

```text
Step-2 stage reuse | student_stage=stem | teacher_module=stem | status=channel_subset_reused | direct=first_norm,second_conv,second_norm | resized=first_conv | unmapped=none
Step-2 stage reuse | student_stage=down1 | teacher_module=layer1 | status=channel_subset_reused | direct=first_conv,first_norm | resized=none | unmapped=second_conv,second_norm
Step-2 head reuse | status=copied | mode=direct | reason=
```

Metadata này cũng được lưu trong `weight_transfer` của phase pruning để có thể xem lại sau khi train:

- `stage_transfer_rows[*].direct_components`
- `stage_transfer_rows[*].resized_components`
- `stage_transfer_rows[*].unmapped_components`
- `head_transfer`
- `exact_match_copy_ratio`

## 7. Step 3: Student training / compression training

Step 3 hiện bám theo logic implementation sau:

1. load blueprint và pruned student checkpoint từ `2_pruning`
2. build `PDGUNet` từ đúng `channel_config` của blueprint
3. load teacher đã freeze từ `1_teacher`
4. train student với:
   `Ltotal = Lseg + lambda_distill * Ldistill + lambda_sparsity * Lsparsity`

Student ở đầu step 3 không khởi tạo random:

- weight ban đầu được load từ checkpoint của `2_pruning`
- checkpoint này đã chứa pruned-student sau bước structural pruning và teacher-weight subset reuse ở step 2

Nếu late hard pruning xảy ra ở cuối step 3:

- compact student mới cũng không khởi tạo random
- repo rebuild kiến trúc nhỏ hơn rồi copy lại subset weight từ student ngay trước thời điểm hard pruning
- distillation sau đó tiếp tục chạy trên compact student đã được kế thừa weight

Nếu `student_variant` bật distillation, teacher sẽ supervise student xuyên suốt toàn bộ step 3. `step3_pruning_epochs` hiện được dùng như cửa sổ epoch cuối để kích hoạt `late hard pruning`; `warmup_pruning_epochs` vẫn còn như alias tương thích CLI cũ. Nếu truyền `--enable_step3_pruning 0`, step 3 sẽ không dùng gate sparsity pressure và không late hard prune.

Trong `3_student/configs/`, repo hiện lưu thêm `student_pruning_config.json` để nhìn nhanh các mốc epoch của pruning, ví dụ:

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

Quy ước hiện tại:

- `step3_pruning_enabled` cho biết step 3 có bật pruning hay không
- `step3_pruning_epochs` là số epoch cuối dành cho late hard pruning khi step 3 pruning bật
- `requested_warmup_pruning_epochs` là giá trị bạn truyền từ CLI hoặc giá trị được map từ `step3_pruning_epochs`
- `warmup_pruning_epochs` là giá trị effective thật sau khi đã clamp theo `max_epochs_student`
- `hard_pruning_apply_epoch` là mốc thực thi hard pruning theo **0-based index**
- nếu số epoch train nhỏ hơn số epoch pruning cuối yêu cầu, config sẽ phản ánh đúng giá trị effective thực tế thay vì chỉ lặp lại giá trị request ban đầu

### Distillation

Hiện tại distillation target là `logits`, chưa phải feature adapter riêng.

### Gating và soft pruning

Step 3 hỗ trợ `learnable channel gating` trên student.

Soft pruning được định nghĩa là:

- gate học được theo channel
- cộng thêm sparsity regularization
- channel nào có gate thấp dần sẽ được xem là bị suy yếu / gần tắt

Hard pruning trong step 3 hiện được thực hiện ở cửa sổ epoch cuối:

- gate được dùng để quyết định channel nào bị giữ / bị cắt
- student được rebuild với `channel_config` nhỏ hơn
- compact student không dùng random init, mà kế thừa subset weight từ student trước prune
- sau đó compact student tiếp tục được distill trong các epoch còn lại

### Late hard pruning policy

Step 3 hiện dùng `step3_pruning_epochs` như số epoch cuối dành cho `late hard pruning`. `warmup_pruning_epochs` vẫn được hỗ trợ để không phá CLI cũ.

Flag chính:

- `--enable_step3_pruning`
- `--step3_pruning_epochs`
- `--warmup_pruning_epochs`
- `--student_variant`
- `--lambda_distill`
- `--lambda_sparsity`
- `--student_gate_near_off_threshold`
- `--student_hard_gate_threshold`

Ý nghĩa:

- nếu `--enable_step3_pruning 0`:
  - không có soft pruning ở step 3
  - không có late hard pruning ở step 3
  - distillation vẫn chạy nếu variant bật distillation
- trong các epoch trước cửa sổ cuối:
  - gating + sparsity active
  - soft pruning behavior được theo dõi
  - nếu variant có distillation thì teacher distillation cũng chạy xuyên suốt
- ở đầu cửa sổ `warmup_pruning_epochs` cuối:
  - hệ thống dùng `student_hard_gate_threshold` hoặc fallback sang `student_gate_near_off_threshold`
  - channel có gate thấp bị cắt cứng thật
  - nếu threshold vẫn giữ toàn bộ channel ở một stage, repo sẽ cắt channel yếu nhất của stage đó để bảo đảm có structural pruning thật
  - student được rebuild với số channel mới
- trong các epoch cuối sau khi cắt:
  - compact student tiếp tục distill từ frozen teacher
  - gate không còn tiếp tục đẩy sparsity như giai đoạn trước

### Student variants

Step 3 hỗ trợ 4 variant:

- `pruned_no_gate`
- `pruned_distill`
- `pruned_gate_sparsity`
- `full`

## 8. Checkpoint và metadata

Checkpoint được quản lý qua `utils/checkpoints.py`.

Mỗi checkpoint có thể lưu:

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

Với proposal branch, `model_info` hiện cũng ghi rõ checkpoint có phải random init hay không, ví dụ:

- `checkpoint_is_random_init: false`
- `checkpoint_weight_status`
- `checkpoint_weight_source`

Điều này đặc biệt hữu ích để kiểm tra:

- `2_pruning` đang lưu model đã reuse weight từ teacher
- `3_student` đang lưu model load từ `2_pruning`
- compact student sau late hard pruning đang lưu model đã reuse subset weight từ student trước prune

Mặc định chỉ lưu:

- `best.pth`
- `last.pth`

Nếu muốn giữ thêm checkpoint history:

```bash
--save_history_checkpoints 1
```

## 9. Output structure

### 9.1 Basic branch

```text
outputs/<model_name>/<dataset>/
├─ artifacts/
│  ├─ channel_analysis/
│  └─ visualizations/
├─ checkpoints/
│  ├─ best.pth
│  ├─ last.pth
│  └─ metadata/
├─ configs/
│  ├─ run_config.json
│  ├─ hyperparameters.json
│  └─ model_config.json
├─ evaluations/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ metrics/
│  └─ basic_metrics.csv
├─ reports/
│  ├─ basic_loss.pdf
│  ├─ basic_performance.pdf
│  ├─ basic_train_visualizations.pdf
│  ├─ basic_val_visualizations.pdf
│  └─ basic_test_visualizations.pdf
└─ run.log
```

### 9.2 Proposal branch

Nếu không truyền `--output_root`, output mặc định nằm dưới `outputs/`. Nếu chạy qua script dataset, `--output_root` là root chung, mặc định `outputs`; cấu hình strategy và step 3 nằm ở folder con như `output_static_0.5_4`, `output_static_0.5_no`, `output_kneedle_auto_4`, `output_middle_static_0.5_4`, `output_middle_kneedle_auto_4`, `output_middle_otsu_auto_4`, `output_middle_gmm_auto_4`, hoặc `output_gmm_auto_no`.

Teacher có root riêng qua `--teacher_output_root`. Khi dùng script dataset, giá trị mặc định là `outputs`, vì vậy:

- `1_teacher` dùng chung: `outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher/`
- `2_pruning`, `3_student`, `pipeline` theo từng thử nghiệm: `<output_root>/pgd_unet/<dataset>/<teacher_model>_teacher/<experiment_folder>/...`

```text
<output_root>/pgd_unet/<dataset>/<teacher_model>_teacher/
├─ 1_teacher/
├─ output_static_0.5_4/
│  ├─ 2_pruning/
│  ├─ 3_student/
│  ├─ student_final/
│  └─ pipeline/
└─ output_kneedle_auto_4/
   ├─ 2_pruning/
   ├─ 3_student/
   ├─ student_final/
   └─ pipeline/
└─ output_middle_static_0.5_4/
   ├─ 2_pruning/
   ├─ 3_student/
   ├─ student_final/
   └─ pipeline/
└─ output_middle_kneedle_auto_4/
   ├─ 2_pruning/
   ├─ 3_student/
   ├─ student_final/
   └─ pipeline/
└─ output_middle_otsu_auto_4/
   ├─ 2_pruning/
   ├─ 3_student/
   ├─ student_final/
   └─ pipeline/
└─ output_middle_gmm_auto_4/
   ├─ 2_pruning/
   ├─ 3_student/
   ├─ student_final/
   └─ pipeline/
```

#### `1_teacher`

Lưu:

- checkpoint teacher
- configs
- metrics
- reports
- evaluations
- channel analysis
- visualizations

#### `2_pruning`

Lưu:

- `blueprint.json`
- pruning summary
- teacher vs student channel comparison
- pruning analysis
- pruned-student baseline checkpoint
- pruned-student evaluation `train/val/test`
- metadata về `weight_transfer`

Quan trọng:

- `2_pruning` bây giờ không chỉ là stage kiến trúc
- nó còn evaluate một `pruned student before tuning` để bạn nhìn được hiệu quả ngay sau pruning
- pruned student này được khởi tạo bằng các weight của teacher tương ứng với những channel không bị prune, thay vì để toàn bộ trọng số ở trạng thái khởi tạo mới
- `weight_transfer` hiện cũng ghi rõ stage nào `direct copy`, stage nào `resized kernel`, stage nào `unmapped`, và `head` có reuse được hay không

#### `3_student`

Lưu:

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

Shortcut export cho weight cuối:

```text
student_final/
├─ best_student.pth
├─ last_student.pth
├─ best_student.json
└─ last_student.json
```

#### `pipeline`

Lưu report tổng hợp toàn pipeline:

- `metrics/pipeline_stage_overview.csv`
- `metrics/pipeline_metrics.csv`
- `metrics/pipeline_compression_summary.csv`
- `reports/pipeline_performance.pdf`
- `reports/pipeline_<phase>_<split>_visualizations.pdf`
- `evaluations/pipeline_summary.json`
- `evaluations/pipeline_summary.md`

Pipeline tổng hợp cả:

- teacher metrics
- pruning baseline metrics
- final student metrics
- compression summary

## 10. Evaluation và metrics

Metric chính:

- Dice
- IoU
- HD95

Thông tin efficiency:

- Params
- FLOPs
- FPS
- inference time

Mỗi phase có thể export:

- `case_metrics.csv`
- `summary.json`
- `summary.md`
- `metrics_summary.json`

Các split cuối mặc định:

- `train`
- `val`
- `test`

## 11. Reports và artifacts

### PDF reports

Repo hiện hỗ trợ:

- `loss.pdf`
- `visualizations.pdf`
- `performance.pdf`
- `channel_analysis_tables.pdf`
- `channel_analysis_charts.pdf`
- `gating_analysis_tables.pdf`
- `gating_analysis_charts.pdf`
- `student_channel_gating_report_tables.pdf`
- `student_channel_gating_report_charts.pdf`

### Channel / gating / pruning artifacts

Các nhóm artifact chính:

- `artifacts/channel_analysis/`
- `artifacts/visualizations/`
- `artifacts/gating_analysis/`
- `artifacts/student_tuning_analysis/`
- `artifacts/pruning_analysis/`

### Student step-3 diagnostics

Step 3 export thêm:

- student input channel profile
- student final channel profile
- gate summary
- gate values per channel
- so sánh `student_input -> student_final`
- per-epoch diagnostics để chứng minh soft pruning behavior

## 12. Một số lưu ý thực tế

- `net_factory.py` không route `pdg_unet`; đây là chủ đích để tách `basic` và `proposal`
- `thop` có thể gắn `total_ops/total_params` vào model; repo đã strip các key này khi save/load checkpoint
- path trong metadata ưu tiên dạng tương đối như `outputs/...` hoặc `evaluations/...`, không lưu absolute path
- `test2d.py` ưu tiên output mới trong `outputs/...`, nhưng vẫn fallback được cho run legacy nếu cần
- distillation của step 3 hiện là `logits distillation`
- late hard pruning của step 3 hiện rebuild lại student thật dựa trên gate threshold ở cửa sổ epoch cuối

## 13. compare_artifacts.py

Repo có thêm `compare_artifacts.py` để gom kết quả:

- `basic baseline`
- `teacher`
- `pruned student`
- `tuned student`

thành một report thống nhất.

### Ví dụ dùng pipeline summary

```bash
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --pipeline_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/pipeline --comparison_name <report_name>
```

### Ví dụ chỉ định từng phase

```bash
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --teacher_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/1_teacher --pruning_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/2_pruning --student_run_dir outputs/pgd_unet/<dataset>/<teacher_model>_teacher/3_student --output_dir outputs/comparisons/<report_name>
```

Script sẽ sinh:

- `stage_overview.csv`
- `performance_comparison.csv`
- `pruning_global_summary.csv`
- `teacher_vs_student_channels.csv`
- `student_tuning_comparison.csv`
- `comparison_summary.json`
- `<comparison_name>.pdf`

## 14. Quick start

1. Cài dependency
2. Chuẩn bị dataset và split
3. Train baseline bằng `train_basic_model.py`
4. Train proposal bằng `train_pgd.py`
5. Đọc output trong `outputs/`
6. So sánh kết quả bằng `compare_artifacts.py`
