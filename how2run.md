# Hướng dẫn chạy project trên Slurm server với Conda

Tài liệu này mô tả chi tiết từng bước để thiết lập môi trường `conda`, viết file `run.sh`, submit job lên Slurm, theo dõi log, và xử lý các lỗi thường gặp. Nội dung được viết dựa trên đúng bối cảnh hiện tại của bạn:

* Server dùng **Slurm**
* Không có sẵn `anaconda` / `miniconda` trong `module avail`
* Bạn đã cài Miniconda tại đường dẫn:

  * `$HOME/miniconda3`
* Project nằm ở:

  * `$HOME/PGD-UNet`
* Environment muốn dùng là:

  * `pgdunet`

---

# 1. Kiểm tra môi trường server

Trước tiên, cần xác định server có những module nào khả dụng.

```bash
module avail
```

Mục đích của lệnh này là để xem server có sẵn:

* Python
* CUDA
* cuDNN
* Anaconda / Miniconda
* Các compiler liên quan

Trong trường hợp của bạn, `module avail` không có `anaconda` hay `miniconda`, vì vậy cần tự cài Miniconda vào thư mục home.

---

# 2. Tải và cài Miniconda

## 2.1. Tải bộ cài Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Lệnh này tải file cài đặt Miniconda cho Linux 64-bit về thư mục hiện tại.

## 2.2. Cài Miniconda vào thư mục home

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

Giải thích:

* `bash Miniconda3-latest-Linux-x86_64.sh`: chạy bộ cài
* `-b`: cài đặt ở chế độ batch, không hỏi tương tác
* `-p $HOME/miniconda3`: chỉ định thư mục cài đặt là `~/miniconda3`

Sau khi cài xong, bạn sẽ có Conda nằm trong:

```bash
$HOME/miniconda3
```

---

# 3. Nạp Conda vào phiên shell hiện tại

Sau khi cài xong, cần nạp script khởi tạo Conda để shell hiện tại nhận lệnh `conda`.

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

Kiểm tra lại:

```bash
conda --version
```

Nếu thấy kết quả như:

```bash
conda 26.1.1
```

thì có nghĩa là Conda đã hoạt động.

---

# 4. Chấp nhận Terms of Service của Conda channels

Ở bản Conda mới, khi dùng các channel mặc định của Anaconda lần đầu, bạn cần chấp nhận điều khoản sử dụng.

Chạy lần lượt 2 lệnh sau:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Nếu không chạy bước này, lệnh `conda create` sẽ báo lỗi và không tạo được environment.

---

# 5. Tạo môi trường Conda

Tạo environment tên `pgdunet` với Python 3.10:

```bash
conda create -n pgdunet python=3.10 -y
```

Giải thích:

* `-n pgdunet`: đặt tên môi trường là `pgdunet`
* `python=3.10`: chọn phiên bản Python 3.10
* `-y`: tự động đồng ý với các câu hỏi xác nhận

Sau đó activate môi trường:

```bash
conda activate pgdunet
```

Kiểm tra:

```bash
conda info --envs
which python
python --version
```

Nếu prompt hiện thêm `(pgdunet)` ở đầu dòng thì có nghĩa là environment đã được kích hoạt.

---

# 6. Cài các package cần thiết

Tùy theo project, bạn cần cài các thư viện Python cần thiết. Với project segmentation bằng PyTorch, có thể bắt đầu với các lệnh sau.

## 6.1. Cài PyTorch bản GPU

Server của bạn có module CUDA 11.8, vì vậy có thể dùng wheel PyTorch cho `cu118`.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 6.2. Cài các package cơ bản khác

```bash
pip install numpy pandas scikit-learn tqdm matplotlib opencv-python
```

Nếu project có yêu cầu khác, bạn có thể cài thêm theo file `requirements.txt` nếu có:

```bash
pip install -r requirements.txt
```

---

# 7. Kiểm tra PyTorch có nhận GPU hay không

Sau khi cài xong, kiểm tra nhanh bằng lệnh:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Nếu dòng thứ hai in ra:

```bash
True
```

thì PyTorch đã nhận GPU.

Nếu ra `False` thì có thể do:

* chưa load CUDA module trong job
* cài sai bản PyTorch
* đang đứng ở login node không có GPU

Lưu ý: việc `torch.cuda.is_available()` ra `False` trên **login node** chưa chắc là lỗi, vì nhiều cluster chỉ có GPU trên compute node.

---

# 8. Cấu trúc cơ bản của một file Slurm `run.sh`

Một file Slurm script thường gồm 3 phần:

1. **Shebang**: xác định shell dùng để chạy script
2. **Các dòng `#SBATCH`**: mô tả tài nguyên cần xin từ Slurm
3. **Các lệnh thực thi**: load module, activate conda, chạy code

Ví dụ khung cơ bản:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

mkdir -p logs

module load cuda-11.8.0-gcc-11.4.0-cuusula
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet

cd $HOME/PGD-UNet

python train.py
```

---

# 9. Giải thích chi tiết từng dòng trong Slurm script

## 9.1. Dòng shebang

```bash
#!/bin/bash
```

Cho biết script sẽ được chạy bằng `bash`.

## 9.2. Tên job

```bash
#SBATCH --job-name=myjob
```

Đặt tên cho job để dễ theo dõi trong `squeue` và log.

## 9.3. File log output và error

```bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Giải thích:

* `%x`: tên job
* `%j`: job ID

Ví dụ file log có thể là:

```bash
logs/myjob_12345.out
logs/myjob_12345.err
```

## 9.4. Số node

```bash
#SBATCH --nodes=1
```

Yêu cầu 1 node, tức là 1 máy vật lý.

Với phần lớn bài toán deep learning thông thường, giá trị này nên để là `1`.

## 9.5. Số task

```bash
#SBATCH --ntasks=1
```

Yêu cầu 1 tiến trình chính.

Đối với các bài train đơn GPU hoặc đơn process, giá trị này thường là `1`.

## 9.6. Số CPU cho mỗi task

```bash
#SBATCH --cpus-per-task=4
```

Cấp phát 4 CPU cores cho tiến trình chính. Thông số này có thể ảnh hưởng đến tốc độ đọc dữ liệu nếu code dùng `num_workers` trong DataLoader.

## 9.7. RAM

```bash
#SBATCH --mem=16G
```

Xin 16 GB RAM cho job.

## 9.8. Thời gian tối đa

```bash
#SBATCH --time=24:00:00
```

Cho phép job chạy tối đa 24 giờ.

## 9.9. GPU

```bash
#SBATCH --gres=gpu:1
```

Yêu cầu 1 GPU.

---

# 10. File `run.sh` cho lệnh train supervised UNet

Đây là file hoàn chỉnh cho lệnh:

```bash
python train_basic_model.py --dataset kvasir --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

## Nội dung file `run_basic_unet.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pgdunet_kvasir
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

mkdir -p logs

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Job Name  : $SLURM_JOB_NAME"
echo "Node      : $(hostname)"
echo "Start time: $(date)"
echo "=============================="

module load cuda-11.8.0-gcc-11.4.0-cuusula
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet

cd $HOME/PGD-UNet

nvidia-smi

python train_basic_model.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model unet \
  --exp supervised_unet \
  --max_epochs 100 \
  --batch_size 8 \
  --base_lr 0.01 \
  --patch_size 256 256

echo "=============================="
echo "End time: $(date)"
echo "=============================="
```

---

# 11. File `run.sh` cho lệnh train PGD

Đây là file hoàn chỉnh cho lệnh:

```bash
python train_pgd.py --dataset cvc --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --exp pgd_cvc --max_epochs_teacher 50 --max_epochs_student 50 --prune_ratio 0.5 --lambda_distill 0.3 --lambda_sparsity 0.3 --batch_size 8 --patch_size 256 256
```

## Nội dung file `run_pgd.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pgd_cvc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

mkdir -p logs

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "=============================="

module load cuda-11.8.0-gcc-11.4.0-cuusula
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet

cd $HOME/PGD-UNet

nvidia-smi

python train_pgd.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --teacher_model unet_resnet152 \
  --exp pgd_cvc \
  --max_epochs_teacher 50 \
  --max_epochs_student 50 \
  --prune_ratio 0.5 \
  --lambda_distill 0.3 \
  --lambda_sparsity 0.3 \
  --batch_size 8 \
  --patch_size 256 256

echo "=============================="
echo "End       : $(date)"
echo "=============================="
```

---

# 12. Cách tạo file `run.sh`

Có nhiều cách, nhưng đơn giản nhất là dùng `nano`.

Ví dụ tạo file `run_pgd.sh`:

```bash
nano run_pgd.sh
```

Dán nội dung script vào.

Sau đó:

* nhấn `Ctrl + O` để lưu
* nhấn `Enter` để xác nhận tên file
* nhấn `Ctrl + X` để thoát

Cấp quyền thực thi:

```bash
chmod +x run_pgd.sh
```

---

# 13. Cách submit job lên Slurm

Sau khi có file `.sh`, submit bằng lệnh:

```bash
chmod +x run_pgd.sh
sbatch run_pgd.sh
```

Nếu thành công, Slurm sẽ trả về dạng:

```bash
Submitted batch job 12345
```

Trong đó `12345` là job ID.

---

# 14. Cách theo dõi job

## 14.1. Xem job đang chờ hoặc đang chạy

```bash
squeue -u $USER
```

Lệnh này hiển thị tất cả job của bạn.

## 14.2. Xem chi tiết job

```bash
scontrol show job 12345
```

Thay `12345` bằng job ID thực tế.

## 14.3. Hủy job

```bash
scancel 12345
```

Lệnh này dùng khi cần dừng job.

---

# 15. Cách xem log

Nếu bạn đã khai báo:

```bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

thì log sẽ nằm trong thư mục `logs`.

Liệt kê file log:

```bash
ls logs
```

Xem log theo thời gian thực:

```bash
tail -f logs/pgd_cvc_12345.out
```

Xem lỗi:

```bash
cat logs/pgd_cvc_12345.err
```

---

# 16. Quy trình chạy đầy đủ từ đầu đến cuối

Dưới đây là quy trình đầy đủ theo đúng thứ tự.

## Bước 1: Nạp Conda

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

## Bước 2: Activate environment

```bash
conda activate pgdunet
```

## Bước 3: Đi vào thư mục project

```bash
cd $HOME/PGD-UNet
```

## Bước 4: Tạo hoặc chỉnh file `run.sh`

```bash
nano run_pgd.sh
```

## Bước 5: Cấp quyền thực thi

```bash
chmod +x run_pgd.sh
```

## Bước 6: Submit job

```bash
sbatch run_pgd.sh
```

## Bước 7: Theo dõi job

```bash
squeue -u $USER
```

## Bước 8: Xem log

```bash
tail -f logs/pgd_cvc_*.out
```

---

# 17. Cách debug trước khi submit job

Trước khi submit, bạn có thể chạy thử trực tiếp bằng `bash` để xem script có lỗi cú pháp hay lỗi môi trường không.

```bash
bash run_pgd.sh
```

Lưu ý: cách này sẽ chạy ngay trên node hiện tại. Nếu bạn đang ở login node thì có thể không phù hợp để train thật, nhưng rất hữu ích để kiểm tra:

* đường dẫn có đúng không
* `conda activate` có hoạt động không
* script có typo không
* tên file Python có đúng không

---

# 18. Các lỗi thường gặp và cách xử lý

## 18.1. `conda: command not found`

Nguyên nhân: chưa source Conda trong shell hoặc trong file `.sh`.

Cách sửa:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

## 18.2. `EnvironmentNameNotFound`

Nguyên nhân: environment chưa được tạo hoặc gõ sai tên.

Kiểm tra:

```bash
conda info --envs
```

## 18.3. `torch.cuda.is_available() = False`

Nguyên nhân có thể là:

* đang ở login node không có GPU
* chưa `module load cuda-11.8.0-gcc-11.4.0-cuusula`
* cài sai bản PyTorch

## 18.4. Job pending mãi

Nguyên nhân có thể là:

* thiếu tài nguyên trống
* xin quá nhiều RAM hoặc GPU
* cluster yêu cầu partition cụ thể

Kiểm tra:

```bash
scontrol show job <jobid>
```

## 18.5. Không có thư mục `logs`

Nếu chưa tạo thư mục `logs`, file output có thể không ghi được đúng như mong muốn.

Cách sửa:

```bash
mkdir -p logs
```

## 18.6. Lỗi do xuống dòng sai trong command

Ví dụ sai:

```bash
--max_epochs_studen
t 50
```

Khi viết command nhiều dòng trong shell script, nên dùng dấu `\` ở cuối dòng trước để nối lệnh cho đúng.

---

# 19. Gợi ý tối ưu thực tế

## 19.1. Tăng số CPU nếu DataLoader dùng nhiều worker

Nếu code có `num_workers > 0`, bạn có thể tăng:

```bash
#SBATCH --cpus-per-task=8
```

## 19.2. Tăng RAM nếu dataset lớn

Ví dụ:

```bash
#SBATCH --mem=32G
```

## 19.3. Tăng thời gian nếu job chạy lâu

Ví dụ cho 48 giờ:

```bash
#SBATCH --time=48:00:00
```

## 19.4. Lưu environment để tái sử dụng

```bash
conda env export > environment.yml
```

Sau này có thể khôi phục bằng:

```bash
conda env create -f environment.yml
```

---

# 20. Checklist ngắn gọn

Dùng checklist này mỗi lần chạy job:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet
cd $HOME/PGD-UNet
mkdir -p logs
sbatch run_pgd.sh
squeue -u $USER
```

Xem log:

```bash
tail -f logs/pgd_cvc_*.out
```

---

# 21. Kết luận

Quy trình chuẩn để chạy project của bạn trên Slurm gồm các bước chính:

1. Cài Miniconda vào thư mục home
2. Source Conda
3. Tạo environment `pgdunet`
4. Cài các package cần thiết
5. Viết file `run.sh` với các dòng `#SBATCH`
6. Load CUDA và activate Conda trong script
7. Submit bằng `sbatch`
8. Theo dõi bằng `squeue` và `tail -f`

Nếu sau này bạn có thêm model mới, bạn chỉ cần sửa phần command Python trong file `run.sh`, còn khung Slurm script gần như có thể giữ nguyên.
