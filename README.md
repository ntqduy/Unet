# Medical Image Segmentation Codebase

Codebase PyTorch cho bài toán medical image segmentation 2D, được tổ chức theo 2 nhánh rõ ràng:

- `basic branch`: train và benchmark các baseline như `unet`, `unet_resnet152`, `resunet`, `vnet`, `unetr`
- `proposal branch`: pipeline cho `pdg_unet` gồm `teacher -> pruning -> student`

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
│  │  ├─ pruning.py
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

- `kvasir`
- `kvasir_seg`
- `cvc`
- `cvc_clinicdb`
- `cyst2d`
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

## 5. Basic branch

### Train

Ví dụ:

```bash
python train_basic_model.py --dataset kvasir --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

Ví dụ với `unet_resnet152`:

```bash
python train_basic_model.py --dataset cvc --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

### Evaluate

```bash
python test2d.py --dataset kvasir --root_path data/Kvasir-SEG --model unet --split test
```

### Checkpoint behavior

`train_basic_model.py` sẽ:

- kiểm tra checkpoint compatible trong `outputs/<model>/<dataset>/checkpoints/`
- nếu có và không bật `--force_retrain 1` thì load lại, skip train, rồi vẫn export final evaluation / reports / artifacts
- nếu chưa có checkpoint phù hợp thì mới train

## 6. Proposal branch: PDG-UNet

### Train full pipeline

```bash
python train_pgd.py --dataset kvasir --root_path data/Kvasir-SEG --teacher_model unet_resnet152 --exp pdg_kvasir --max_epochs_teacher 50 --max_epochs_student 100 --prune_ratio 0.5 --lambda_distill 0.3 --lambda_sparsity 0.3 --batch_size 8 --patch_size 256 256
```

### Teacher reuse

Khi cần teacher checkpoint, `train_pgd.py` sẽ check theo thứ tự:

1. `outputs/pdg_unet/<dataset>/<teacher>_teacher/1_teacher/checkpoints/`
2. explicit `--teacher_checkpoint` nếu có
3. `outputs/<teacher_model>/<dataset>/checkpoints/`

Nếu reuse từ `basic branch`, checkpoint sẽ được register lại vào `1_teacher/checkpoints/` để output proposal luôn đầy đủ và dễ đọc.

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

1. phân tích teacher và chọn `kept_channel_indices` theo pruning criterion
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

## 7. Step 3: Student training / compression training

Step 3 hiện bám theo logic implementation sau:

1. load blueprint và pruned student checkpoint từ `2_pruning`
2. build `PDGUNet` từ đúng `channel_config` của blueprint
3. load teacher đã freeze từ `1_teacher`
4. train student với:
   `Ltotal = Lseg + lambda_distill * Ldistill + lambda_sparsity * Lsparsity`

Nếu `student_variant` bật distillation, teacher sẽ supervise student xuyên suốt toàn bộ step 3. `warmup_pruning_epochs` hiện được dùng như cửa sổ epoch cuối để kích hoạt `late hard pruning`, chứ không còn là khoảng epoch đầu nữa.

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
- sau đó compact student tiếp tục được distill trong các epoch còn lại

### Late hard pruning policy

Step 3 hiện dùng `warmup_pruning_epochs` như số epoch cuối dành cho `late hard pruning`.

Flag chính:

- `--warmup_pruning_epochs`
- `--student_variant`
- `--lambda_distill`
- `--lambda_sparsity`
- `--student_gate_near_off_threshold`
- `--student_hard_gate_threshold`

Ý nghĩa:

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

```text
outputs/pdg_unet/<dataset>/<teacher_model>_teacher/
├─ 1_teacher/
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

#### `3_student`

Lưu:

- checkpoint student
- configs
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
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --pipeline_dir outputs/pdg_unet/<dataset>/<teacher_model>_teacher/pipeline --comparison_name <report_name>
```

### Ví dụ chỉ định từng phase

```bash
python compare_artifacts.py --basic_run_dir outputs/<basic_model>/<dataset> --teacher_run_dir outputs/pdg_unet/<dataset>/<teacher_model>_teacher/1_teacher --pruning_run_dir outputs/pdg_unet/<dataset>/<teacher_model>_teacher/2_pruning --student_run_dir outputs/pdg_unet/<dataset>/<teacher_model>_teacher/3_student --output_dir outputs/comparisons/<report_name>
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
