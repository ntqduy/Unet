# Medical Image Segmentation Codebase

Codebase huấn luyện và đánh giá phân đoạn ảnh y tế 2D bằng PyTorch, được tổ chức theo 2 nhánh chính:

- `basic branch`: benchmark các mô hình baseline như `UNet`, `UNet_ResNet`, `VNet`, `UNETR`, ...
- `proposal branch`: pipeline đầy đủ cho mô hình đề xuất `pdg_unet` gồm `teacher -> pruning -> student -> distillation/gating`

Mục tiêu của repo là chuẩn hóa interface output, checkpoint, metrics và artifact để có thể so sánh công bằng giữa baseline models và proposal model.

## 1. Tổng quan kiến trúc

### Basic branch

- Kiến trúc model nằm trong `networks/Basic_Model`
- Model router/factory nằm trong `networks/net_factory.py`
- Script train chính là `train_basic_model.py`
- Script evaluate riêng là `test2d.py`

Các model hiện có:

- `unet`
- `unet_resnet152`
- `resunet`
- `vnet`
- `unetr`

### Proposal branch

- Kiến trúc proposal nằm trong `networks/PGD_Unet`
- Student model là `PDGUNet` trong `networks/PGD_Unet/gated_unet.py`
- Pruning utilities nằm trong `networks/PGD_Unet/pruning.py`
- Script train pipeline là `train_pgd.py`

Pipeline proposal gồm 3 phase:

1. Train hoặc load lại `teacher`
2. Structured pruning để sinh `blueprint`
3. Build và train `student` với `segmentation loss + distillation loss + sparsity/gating loss`

## 2. Chuẩn hóa output model

Toàn bộ model hiện được chuẩn hóa về một output chung qua `SegmentationModelOutput` trong `utils/model_output.py`.

Output chuẩn gồm:

- `logits`: tensor segmentation trước sigmoid/softmax
- `probs`: xác suất sau sigmoid/softmax
- `preds`: mask dự đoán để evaluate/inference
- `features`: feature maps trung gian
- `aux`: metadata phụ
- `model_name`
- `backbone_name`
- `student_name`
- `phase_name`

Điểm quan trọng:

- `train_basic_model.py` và `train_pgd.py` có thể dùng lại phần lớn logic evaluate/export
- Không cần viết nhiều `if-else` riêng cho từng model output
- Baseline và proposal được đưa về cùng chuẩn output để so sánh trực tiếp

## 3. Cấu trúc thư mục chính

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
└─ logs/
```

Lưu ý:

- `prunning.py` được giữ lại như alias tương thích ngược cho code cũ
- `net_factory.py` hiện chỉ đóng vai trò router cho `basic branch`

## 4. Yêu cầu môi trường

### Python

- Khuyến nghị `Python 3.10+`

### Cài đặt dependency

```bash
pip install -r requirements.txt
```

Nếu muốn tính FLOPs trong performance report, cài thêm:

```bash
pip install thop
```

## 5. Chuẩn bị dữ liệu

Repo hiện hỗ trợ các dataset key:

- `kvasir`
- `kvasir_seg`
- `cvc`
- `cvc_clinicdb`
- `cyst2d`
- `generic`

Dataset loader nằm trong `dataloaders/dataset.py` và hỗ trợ:

- tự quét cặp `image/mask`
- manifest split `train.txt`, `val.txt`, `test.txt`
- normalize mask nhị phân về `0/1`
- transform train/eval riêng

Ví dụ:

```text
data/
└─ Kvasir-SEG/
   ├─ images/
   ├─ masks/
   ├─ train.txt
   ├─ val.txt
   └─ test.txt
```

## 6. Huấn luyện basic models

### Ví dụ train UNet

```bash
python train_basic_model.py --dataset kvasir --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

### Ví dụ train UNet-ResNet152

```bash
python train_basic_model.py --dataset cvc --root_path data/CVC-ClinicDB --model unet_resnet152 --encoder_pretrained 1 --exp supervised_resnet152 --max_epochs 100
```

### Evaluate checkpoint baseline

```bash
python test2d.py --dataset kvasir --root_path data/Kvasir-SEG --exp supervised_unet --model unet --split test
```

## 7. Huấn luyện proposal model PDG-UNet

### Ví dụ chạy pipeline đầy đủ

```bash
python train_pgd.py --dataset kvasir --root_path data/Kvasir-SEG --teacher_model unet_resnet152 --exp pdg_kvasir --max_epochs_teacher 50 --max_epochs_student 100 --prune_ratio 0.5 --lambda_distill 0.3 --lambda_sparsity 0.3 --batch_size 8 --patch_size 256 256
```

### Hành vi mặc định của pipeline

- Nếu `teacher checkpoint` đã tồn tại trong phase `teacher` thì sẽ load lại
- Nếu `blueprint.json` đã tồn tại trong phase `pruning` thì sẽ load lại
- Nếu `student checkpoint` đã tồn tại trong phase `student` thì sẽ load lại

### Các flag hữu ích

- `--teacher_checkpoint`: chỉ định checkpoint teacher có sẵn
- `--force_retrain_teacher 1`: train lại teacher từ đầu
- `--force_reprune 1`: prune lại dù đã có blueprint
- `--force_retrain_student 1`: train lại student từ đầu

## 8. Checkpoint và khả năng tái sử dụng

Checkpoint được quản lý qua `utils/checkpoints.py`.

Mỗi checkpoint hiện lưu:

- `model_state_dict`
- `optimizer_state_dict` nếu có
- `epoch`
- `global_step`
- `best_metric`
- `metrics`
- `config`
- `model_info`
- `phase`
- `extra_state`

Điều này cho phép:

- resume training
- load lại backbone/teacher/student đúng kiến trúc
- tái sử dụng teacher checkpoint để prune lại
- tái sử dụng student checkpoint để fine-tune hoặc evaluate tiếp
- tái sử dụng blueprint để build lại student mà không cần prune lại

## 9. Artifact output

Repo hiện chuẩn hóa artifact theo các nhóm sau.

### 9.1 Checkpoints

Mỗi run có thư mục:

```text
checkpoints/
├─ best.pth
├─ last.pth
├─ epoch_xxx.pth
└─ metadata/
   ├─ best.json
   ├─ last.json
   └─ epoch_xxx.json
```

### 9.2 Evaluation summaries

Mỗi split evaluate có:

- `summary.json`
- `summary.md`
- `metrics_summary.json`
- `case_metrics.csv`
- ảnh visualization PNG theo thư mục con

### 9.3 CSV metrics

Các CSV tổng hợp được ghi bởi `utils/reporting.py` và có thể chứa:

- `experiment`
- `dataset`
- `split`
- `phase`
- `model_name`
- `backbone_name`
- `student_name`
- `dice`
- `iou`
- `hd95`
- `params`
- `trainable_params`
- `flops`
- `fps`
- `inference_time_seconds`
- `evaluation_time_seconds`
- `checkpoint_path`

### 9.4 PDF reports

Repo hiện hỗ trợ:

- `loss.pdf`
- `visualizations.pdf`
- `performance.pdf`

Trong đó:

- `loss.pdf` dùng để theo dõi loss theo epoch
- `visualizations.pdf` gồm `image / ground truth / prediction`
- `performance.pdf` gồm bảng tổng hợp và biểu đồ các metric chính

## 10. Vị trí output hiện tại

### Basic branch

Để giữ tương thích ngược, basic branch hiện vẫn lưu chính tại:

```text
logs/model/supervised/<exp>/
```

Trong đó sẽ có thêm:

- `checkpoints/`
- `metrics/`
- `reports/`
- `evaluations/`

### Proposal branch

Proposal branch hiện lưu theo phase tại:

```text
logs/runs/proposal/<exp>/<dataset>/pdg_unet/
├─ pipeline/<teacher_model>/
├─ teacher/<teacher_model>/
├─ pruning/<teacher_model>_ratio_<ratio>/
└─ student/<teacher_model>_ratio_<ratio>/
```

## 11. Metric đang dùng

Evaluation hiện đã được chuẩn hóa về:

- Dice
- IoU
- HD95

Metric logic nằm trong:

- `utils/val_2d.py`
- `utils/evaluation.py`

## 12. Loss đang dùng

### Basic branch

`train_basic_model.py` hiện dùng:

- `CrossEntropyLoss`
- `DiceLoss`
- tổng hợp thành `0.5 * (CE + Dice)`

### Proposal branch

`train_pgd.py` hiện dùng `CompressionLoss` gồm:

- `segmentation_loss`
- `distillation_loss`
- `sparsity_loss`
- `total_loss`

## 13. Vai trò của các file chính

- `train_basic_model.py`: train và evaluate các baseline models
- `train_pgd.py`: pipeline teacher -> pruning -> student cho proposal model
- `test2d.py`: evaluate checkpoint baseline độc lập
- `networks/net_factory.py`: router/factory cho basic models
- `utils/model_output.py`: output contract chuẩn cho mọi model
- `utils/checkpoints.py`: save/load checkpoint có metadata
- `utils/reporting.py`: export CSV/PDF dùng chung
- `utils/profiling.py`: params, FLOPs, FPS, inference time

## 14. Một số lưu ý thực tế

- `net_factory.py` hiện chưa route `pdg_unet`; đây là chủ đích để tách rõ basic branch và proposal branch
- FLOPs chỉ được ghi nếu bạn cài thêm `thop`
- Basic branch và proposal branch đã dùng chung chuẩn metrics/report, nhưng đường dẫn output gốc vẫn đang khác nhau để giữ tương thích code cũ
- Nếu muốn migration hoàn toàn sang một output root duy nhất, có thể refactor tiếp ở bước sau

## 15. Hướng phát triển tiếp theo

Một số bước nên làm tiếp nếu muốn codebase sạch hơn nữa:

- tách trainer/evaluator/exporter thành module riêng thay vì để nhiều logic trong script train
- gom basic branch sang `logs/runs/basic/...` để đồng bộ tuyệt đối với proposal branch
- thêm config YAML hoặc JSON tập trung thay vì phụ thuộc hoàn toàn vào CLI
- thêm `resume` chính thức cho từng phase
- thêm benchmark script tổng hợp nhiều model vào cùng một bảng so sánh

## 16. Tóm tắt nhanh

Nếu bạn chỉ muốn bắt đầu nhanh:

1. Cài dependency
2. Chuẩn bị dataset và split
3. Train baseline bằng `train_basic_model.py`
4. Train proposal bằng `train_pgd.py`
5. So sánh CSV/PDF/checkpoint artifact giữa hai nhánh

Repo hiện đã được tổ chức theo hướng:

- output model thống nhất
- checkpoint rõ ràng
- artifact xuất đồng nhất
- support tái sử dụng teacher/student/blueprint
- dễ so sánh công bằng giữa baseline và proposal
