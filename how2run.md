# HÆ°á»›ng dáº«n cháº¡y project trÃªn Slurm server vá»›i Conda

TÃ i liá»‡u nÃ y mÃ´ táº£ chi tiáº¿t tá»«ng bÆ°á»›c Ä‘á»ƒ thiáº¿t láº­p mÃ´i trÆ°á»ng `conda`, viáº¿t file `run.sh`, submit job lÃªn Slurm, theo dÃµi log, vÃ  xá»­ lÃ½ cÃ¡c lá»—i thÆ°á»ng gáº·p. Ná»™i dung Ä‘Æ°á»£c viáº¿t dá»±a trÃªn Ä‘Ãºng bá»‘i cáº£nh hiá»‡n táº¡i cá»§a báº¡n:

* Server dÃ¹ng **Slurm**
* KhÃ´ng cÃ³ sáºµn `anaconda` / `miniconda` trong `module avail`
* Báº¡n Ä‘Ã£ cÃ i Miniconda táº¡i Ä‘Æ°á»ng dáº«n:

  * `$HOME/miniconda3`
* Project náº±m á»Ÿ:

  * `$HOME/PGD-UNet`
* Environment muá»‘n dÃ¹ng lÃ :

  * `pgdunet`

---

# 1. Kiá»ƒm tra mÃ´i trÆ°á»ng server

TrÆ°á»›c tiÃªn, cáº§n xÃ¡c Ä‘á»‹nh server cÃ³ nhá»¯ng module nÃ o kháº£ dá»¥ng.

```bash
module avail
```

Má»¥c Ä‘Ã­ch cá»§a lá»‡nh nÃ y lÃ  Ä‘á»ƒ xem server cÃ³ sáºµn:

* Python
* CUDA
* cuDNN
* Anaconda / Miniconda
* CÃ¡c compiler liÃªn quan

Trong trÆ°á»ng há»£p cá»§a báº¡n, `module avail` khÃ´ng cÃ³ `anaconda` hay `miniconda`, vÃ¬ váº­y cáº§n tá»± cÃ i Miniconda vÃ o thÆ° má»¥c home.

---

# 2. Táº£i vÃ  cÃ i Miniconda

## 2.1. Táº£i bá»™ cÃ i Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Lá»‡nh nÃ y táº£i file cÃ i Ä‘áº·t Miniconda cho Linux 64-bit vá» thÆ° má»¥c hiá»‡n táº¡i.

## 2.2. CÃ i Miniconda vÃ o thÆ° má»¥c home

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

Giáº£i thÃ­ch:

* `bash Miniconda3-latest-Linux-x86_64.sh`: cháº¡y bá»™ cÃ i
* `-b`: cÃ i Ä‘áº·t á»Ÿ cháº¿ Ä‘á»™ batch, khÃ´ng há»i tÆ°Æ¡ng tÃ¡c
* `-p $HOME/miniconda3`: chá»‰ Ä‘á»‹nh thÆ° má»¥c cÃ i Ä‘áº·t lÃ  `~/miniconda3`

Sau khi cÃ i xong, báº¡n sáº½ cÃ³ Conda náº±m trong:

```bash
$HOME/miniconda3
```

---

# 3. Náº¡p Conda vÃ o phiÃªn shell hiá»‡n táº¡i

Sau khi cÃ i xong, cáº§n náº¡p script khá»Ÿi táº¡o Conda Ä‘á»ƒ shell hiá»‡n táº¡i nháº­n lá»‡nh `conda`.

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

Kiá»ƒm tra láº¡i:

```bash
conda --version
```

Náº¿u tháº¥y káº¿t quáº£ nhÆ°:

```bash
conda 26.1.1
```

thÃ¬ cÃ³ nghÄ©a lÃ  Conda Ä‘Ã£ hoáº¡t Ä‘á»™ng.

---

# 4. Cháº¥p nháº­n Terms of Service cá»§a Conda channels

á»ž báº£n Conda má»›i, khi dÃ¹ng cÃ¡c channel máº·c Ä‘á»‹nh cá»§a Anaconda láº§n Ä‘áº§u, báº¡n cáº§n cháº¥p nháº­n Ä‘iá»u khoáº£n sá»­ dá»¥ng.

Cháº¡y láº§n lÆ°á»£t 2 lá»‡nh sau:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Náº¿u khÃ´ng cháº¡y bÆ°á»›c nÃ y, lá»‡nh `conda create` sáº½ bÃ¡o lá»—i vÃ  khÃ´ng táº¡o Ä‘Æ°á»£c environment.

---

# 5. Táº¡o mÃ´i trÆ°á»ng Conda

Táº¡o environment tÃªn `pgdunet` vá»›i Python 3.10:

```bash
conda create -n pgdunet python=3.10 -y
```

Giáº£i thÃ­ch:

* `-n pgdunet`: Ä‘áº·t tÃªn mÃ´i trÆ°á»ng lÃ  `pgdunet`
* `python=3.10`: chá»n phiÃªn báº£n Python 3.10
* `-y`: tá»± Ä‘á»™ng Ä‘á»“ng Ã½ vá»›i cÃ¡c cÃ¢u há»i xÃ¡c nháº­n

Sau Ä‘Ã³ activate mÃ´i trÆ°á»ng:

```bash
conda activate pgdunet
```

Kiá»ƒm tra:

```bash
conda info --envs
which python
python --version
```

Náº¿u prompt hiá»‡n thÃªm `(pgdunet)` á»Ÿ Ä‘áº§u dÃ²ng thÃ¬ cÃ³ nghÄ©a lÃ  environment Ä‘Ã£ Ä‘Æ°á»£c kÃ­ch hoáº¡t.

---

# 6. CÃ i cÃ¡c package cáº§n thiáº¿t

TÃ¹y theo project, báº¡n cáº§n cÃ i cÃ¡c thÆ° viá»‡n Python cáº§n thiáº¿t. Vá»›i project segmentation báº±ng PyTorch, cÃ³ thá»ƒ báº¯t Ä‘áº§u vá»›i cÃ¡c lá»‡nh sau.

## 6.1. CÃ i PyTorch báº£n GPU

Server cá»§a báº¡n cÃ³ module CUDA 11.8, vÃ¬ váº­y cÃ³ thá»ƒ dÃ¹ng wheel PyTorch cho `cu118`.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 6.2. CÃ i cÃ¡c package cÆ¡ báº£n khÃ¡c

```bash
pip install numpy pandas scikit-learn tqdm matplotlib opencv-python
```

Náº¿u project cÃ³ yÃªu cáº§u khÃ¡c, báº¡n cÃ³ thá»ƒ cÃ i thÃªm theo file `requirements.txt` náº¿u cÃ³:

```bash
pip install -r requirements.txt
```

---

# 7. Kiá»ƒm tra PyTorch cÃ³ nháº­n GPU hay khÃ´ng

Sau khi cÃ i xong, kiá»ƒm tra nhanh báº±ng lá»‡nh:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Náº¿u dÃ²ng thá»© hai in ra:

```bash
True
```

thÃ¬ PyTorch Ä‘Ã£ nháº­n GPU.

Náº¿u ra `False` thÃ¬ cÃ³ thá»ƒ do:

* chÆ°a load CUDA module trong job
* cÃ i sai báº£n PyTorch
* Ä‘ang Ä‘á»©ng á»Ÿ login node khÃ´ng cÃ³ GPU

LÆ°u Ã½: viá»‡c `torch.cuda.is_available()` ra `False` trÃªn **login node** chÆ°a cháº¯c lÃ  lá»—i, vÃ¬ nhiá»u cluster chá»‰ cÃ³ GPU trÃªn compute node.

---

# 8. Cáº¥u trÃºc cÆ¡ báº£n cá»§a má»™t file Slurm `run.sh`

Má»™t file Slurm script thÆ°á»ng gá»“m 3 pháº§n:

1. **Shebang**: xÃ¡c Ä‘á»‹nh shell dÃ¹ng Ä‘á»ƒ cháº¡y script
2. **CÃ¡c dÃ²ng `#SBATCH`**: mÃ´ táº£ tÃ i nguyÃªn cáº§n xin tá»« Slurm
3. **CÃ¡c lá»‡nh thá»±c thi**: load module, activate conda, cháº¡y code

VÃ­ dá»¥ khung cÆ¡ báº£n:

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

# 9. Giáº£i thÃ­ch chi tiáº¿t tá»«ng dÃ²ng trong Slurm script

## 9.1. DÃ²ng shebang

```bash
#!/bin/bash
```

Cho biáº¿t script sáº½ Ä‘Æ°á»£c cháº¡y báº±ng `bash`.

## 9.2. TÃªn job

```bash
#SBATCH --job-name=myjob
```

Äáº·t tÃªn cho job Ä‘á»ƒ dá»… theo dÃµi trong `squeue` vÃ  log.

## 9.3. File log output vÃ  error

```bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Giáº£i thÃ­ch:

* `%x`: tÃªn job
* `%j`: job ID

VÃ­ dá»¥ file log cÃ³ thá»ƒ lÃ :

```bash
logs/myjob_12345.out
logs/myjob_12345.err
```

## 9.4. Sá»‘ node

```bash
#SBATCH --nodes=1
```

YÃªu cáº§u 1 node, tá»©c lÃ  1 mÃ¡y váº­t lÃ½.

Vá»›i pháº§n lá»›n bÃ i toÃ¡n deep learning thÃ´ng thÆ°á»ng, giÃ¡ trá»‹ nÃ y nÃªn Ä‘á»ƒ lÃ  `1`.

## 9.5. Sá»‘ task

```bash
#SBATCH --ntasks=1
```

YÃªu cáº§u 1 tiáº¿n trÃ¬nh chÃ­nh.

Äá»‘i vá»›i cÃ¡c bÃ i train Ä‘Æ¡n GPU hoáº·c Ä‘Æ¡n process, giÃ¡ trá»‹ nÃ y thÆ°á»ng lÃ  `1`.

## 9.6. Sá»‘ CPU cho má»—i task

```bash
#SBATCH --cpus-per-task=4
```

Cáº¥p phÃ¡t 4 CPU cores cho tiáº¿n trÃ¬nh chÃ­nh. ThÃ´ng sá»‘ nÃ y cÃ³ thá»ƒ áº£nh hÆ°á»Ÿng Ä‘áº¿n tá»‘c Ä‘á»™ Ä‘á»c dá»¯ liá»‡u náº¿u code dÃ¹ng `num_workers` trong DataLoader.

## 9.7. RAM

```bash
#SBATCH --mem=16G
```

Xin 16 GB RAM cho job.

## 9.8. Thá»i gian tá»‘i Ä‘a

```bash
#SBATCH --time=24:00:00
```

Cho phÃ©p job cháº¡y tá»‘i Ä‘a 24 giá».

## 9.9. GPU

```bash
#SBATCH --gres=gpu:1
```

YÃªu cáº§u 1 GPU.

---

# 10. File `run.sh` cho lá»‡nh train supervised UNet

ÄÃ¢y lÃ  file hoÃ n chá»‰nh cho lá»‡nh:

```bash
python train_basic_model.py --dataset kvasir_seg --root_path data/Kvasir-SEG --model unet --exp supervised_unet --max_epochs 100 --batch_size 8 --base_lr 0.01 --patch_size 256 256
```

## Ná»™i dung file `run_basic_unet.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pgdunet_kvasir_seg
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
  --dataset kvasir_seg \
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

# 11. File `run.sh` cho lá»‡nh train PGD

ÄÃ¢y lÃ  file hoÃ n chá»‰nh cho lá»‡nh:

```bash
python train_pgd.py --dataset cvc_clinicdb --root_path data/CVC-ClinicDB --teacher_model unet_resnet152 --exp pgd_cvc_clinicdb --max_epochs_teacher 50 --max_epochs_student 50 --prune_ratio 0.5 --lambda_distill 0.3 --lambda_sparsity 0.3 --batch_size 8 --patch_size 256 256
```

## Ná»™i dung file `run_pgd_cvc_clinicdb.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pgd_cvc_clinicdb
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
  --dataset cvc_clinicdb \
  --root_path data/CVC-ClinicDB \
  --teacher_model unet_resnet152 \
  --exp pgd_cvc_clinicdb \
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

# 12. CÃ¡ch táº¡o file `run.sh`

CÃ³ nhiá»u cÃ¡ch, nhÆ°ng Ä‘Æ¡n giáº£n nháº¥t lÃ  dÃ¹ng `nano`.

VÃ­ dá»¥ táº¡o file `run_pgd_cvc_clinicdb.sh`:

```bash
nano run_pgd_cvc_clinicdb.sh
```

DÃ¡n ná»™i dung script vÃ o.

Sau Ä‘Ã³:

* nháº¥n `Ctrl + O` Ä‘á»ƒ lÆ°u
* nháº¥n `Enter` Ä‘á»ƒ xÃ¡c nháº­n tÃªn file
* nháº¥n `Ctrl + X` Ä‘á»ƒ thoÃ¡t

Cáº¥p quyá»n thá»±c thi:

```bash
chmod +x run_pgd_cvc_clinicdb.sh
```

---

# 13. CÃ¡ch submit job lÃªn Slurm

Sau khi cÃ³ file `.sh`, submit báº±ng lá»‡nh:

```bash
chmod +x run_pgd_cvc_clinicdb.sh
sbatch run_pgd_cvc_clinicdb.sh
```

Náº¿u thÃ nh cÃ´ng, Slurm sáº½ tráº£ vá» dáº¡ng:

```bash
Submitted batch job 12345
```

Trong Ä‘Ã³ `12345` lÃ  job ID.

---

# 14. CÃ¡ch theo dÃµi job

## 14.1. Xem job Ä‘ang chá» hoáº·c Ä‘ang cháº¡y

```bash
squeue -u $USER
```

Lá»‡nh nÃ y hiá»ƒn thá»‹ táº¥t cáº£ job cá»§a báº¡n.

## 14.2. Xem chi tiáº¿t job

```bash
scontrol show job 12345
```

Thay `12345` báº±ng job ID thá»±c táº¿.

## 14.3. Há»§y job

```bash
scancel 12345
```

Lá»‡nh nÃ y dÃ¹ng khi cáº§n dá»«ng job.

---

# 15. CÃ¡ch xem log

Náº¿u báº¡n Ä‘Ã£ khai bÃ¡o:

```bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

thÃ¬ log sáº½ náº±m trong thÆ° má»¥c `logs`.

Liá»‡t kÃª file log:

```bash
ls logs
```

Xem log theo thá»i gian thá»±c:

```bash
tail -f logs/pgd_cvc_clinicdb_12345.out
```

Xem lá»—i:

```bash
cat logs/pgd_cvc_clinicdb_12345.err
```

---

# 16. Quy trÃ¬nh cháº¡y Ä‘áº§y Ä‘á»§ tá»« Ä‘áº§u Ä‘áº¿n cuá»‘i

DÆ°á»›i Ä‘Ã¢y lÃ  quy trÃ¬nh Ä‘áº§y Ä‘á»§ theo Ä‘Ãºng thá»© tá»±.

## BÆ°á»›c 1: Náº¡p Conda

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

## BÆ°á»›c 2: Activate environment

```bash
conda activate pgdunet
```

## BÆ°á»›c 3: Äi vÃ o thÆ° má»¥c project

```bash
cd $HOME/PGD-UNet
```

## BÆ°á»›c 4: Táº¡o hoáº·c chá»‰nh file `run.sh`

```bash
nano run_pgd_cvc_clinicdb.sh
```

## BÆ°á»›c 5: Cáº¥p quyá»n thá»±c thi

```bash
chmod +x run_pgd_cvc_clinicdb.sh
```

## BÆ°á»›c 6: Submit job

```bash
sbatch run_pgd_cvc_clinicdb.sh
```

## BÆ°á»›c 7: Theo dÃµi job

```bash
squeue -u $USER
```

## BÆ°á»›c 8: Xem log

```bash
tail -f logs/pgd_cvc_clinicdb_*.out
```

---

# 17. CÃ¡ch debug trÆ°á»›c khi submit job

TrÆ°á»›c khi submit, báº¡n cÃ³ thá»ƒ cháº¡y thá»­ trá»±c tiáº¿p báº±ng `bash` Ä‘á»ƒ xem script cÃ³ lá»—i cÃº phÃ¡p hay lá»—i mÃ´i trÆ°á»ng khÃ´ng.

```bash
bash run_pgd_cvc_clinicdb.sh
```

LÆ°u Ã½: cÃ¡ch nÃ y sáº½ cháº¡y ngay trÃªn node hiá»‡n táº¡i. Náº¿u báº¡n Ä‘ang á»Ÿ login node thÃ¬ cÃ³ thá»ƒ khÃ´ng phÃ¹ há»£p Ä‘á»ƒ train tháº­t, nhÆ°ng ráº¥t há»¯u Ã­ch Ä‘á»ƒ kiá»ƒm tra:

* Ä‘Æ°á»ng dáº«n cÃ³ Ä‘Ãºng khÃ´ng
* `conda activate` cÃ³ hoáº¡t Ä‘á»™ng khÃ´ng
* script cÃ³ typo khÃ´ng
* tÃªn file Python cÃ³ Ä‘Ãºng khÃ´ng

---

# 18. CÃ¡c lá»—i thÆ°á»ng gáº·p vÃ  cÃ¡ch xá»­ lÃ½

## 18.1. `conda: command not found`

NguyÃªn nhÃ¢n: chÆ°a source Conda trong shell hoáº·c trong file `.sh`.

CÃ¡ch sá»­a:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

## 18.2. `EnvironmentNameNotFound`

NguyÃªn nhÃ¢n: environment chÆ°a Ä‘Æ°á»£c táº¡o hoáº·c gÃµ sai tÃªn.

Kiá»ƒm tra:

```bash
conda info --envs
```

## 18.3. `torch.cuda.is_available() = False`

NguyÃªn nhÃ¢n cÃ³ thá»ƒ lÃ :

* Ä‘ang á»Ÿ login node khÃ´ng cÃ³ GPU
* chÆ°a `module load cuda-11.8.0-gcc-11.4.0-cuusula`
* cÃ i sai báº£n PyTorch

## 18.4. Job pending mÃ£i

NguyÃªn nhÃ¢n cÃ³ thá»ƒ lÃ :

* thiáº¿u tÃ i nguyÃªn trá»‘ng
* xin quÃ¡ nhiá»u RAM hoáº·c GPU
* cluster yÃªu cáº§u partition cá»¥ thá»ƒ

Kiá»ƒm tra:

```bash
scontrol show job <jobid>
```

## 18.5. KhÃ´ng cÃ³ thÆ° má»¥c `logs`

Náº¿u chÆ°a táº¡o thÆ° má»¥c `logs`, file output cÃ³ thá»ƒ khÃ´ng ghi Ä‘Æ°á»£c Ä‘Ãºng nhÆ° mong muá»‘n.

CÃ¡ch sá»­a:

```bash
mkdir -p logs
```

## 18.6. Lá»—i do xuá»‘ng dÃ²ng sai trong command

VÃ­ dá»¥ sai:

```bash
--max_epochs_studen
t 50
```

Khi viáº¿t command nhiá»u dÃ²ng trong shell script, nÃªn dÃ¹ng dáº¥u `\` á»Ÿ cuá»‘i dÃ²ng trÆ°á»›c Ä‘á»ƒ ná»‘i lá»‡nh cho Ä‘Ãºng.

---

# 19. Gá»£i Ã½ tá»‘i Æ°u thá»±c táº¿

## 19.1. TÄƒng sá»‘ CPU náº¿u DataLoader dÃ¹ng nhiá»u worker

Náº¿u code cÃ³ `num_workers > 0`, báº¡n cÃ³ thá»ƒ tÄƒng:

```bash
#SBATCH --cpus-per-task=8
```

## 19.2. TÄƒng RAM náº¿u dataset lá»›n

VÃ­ dá»¥:

```bash
#SBATCH --mem=32G
```

## 19.3. TÄƒng thá»i gian náº¿u job cháº¡y lÃ¢u

VÃ­ dá»¥ cho 48 giá»:

```bash
#SBATCH --time=48:00:00
```

## 19.4. LÆ°u environment Ä‘á»ƒ tÃ¡i sá»­ dá»¥ng

```bash
conda env export > environment.yml
```

Sau nÃ y cÃ³ thá»ƒ khÃ´i phá»¥c báº±ng:

```bash
conda env create -f environment.yml
```

---

# 20. Checklist ngáº¯n gá»n

DÃ¹ng checklist nÃ y má»—i láº§n cháº¡y job:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet
cd $HOME/PGD-UNet
mkdir -p logs
sbatch run_pgd_cvc_clinicdb.sh
squeue -u $USER
```

Xem log:

```bash
tail -f logs/pgd_cvc_clinicdb_*.out
```

---

# 21. Káº¿t luáº­n

Quy trÃ¬nh chuáº©n Ä‘á»ƒ cháº¡y project cá»§a báº¡n trÃªn Slurm gá»“m cÃ¡c bÆ°á»›c chÃ­nh:

1. CÃ i Miniconda vÃ o thÆ° má»¥c home
2. Source Conda
3. Táº¡o environment `pgdunet`
4. CÃ i cÃ¡c package cáº§n thiáº¿t
5. Viáº¿t file `run.sh` vá»›i cÃ¡c dÃ²ng `#SBATCH`
6. Load CUDA vÃ  activate Conda trong script
7. Submit báº±ng `sbatch`
8. Theo dÃµi báº±ng `squeue` vÃ  `tail -f`

Náº¿u sau nÃ y báº¡n cÃ³ thÃªm model má»›i, báº¡n chá»‰ cáº§n sá»­a pháº§n command Python trong file `run.sh`, cÃ²n khung Slurm script gáº§n nhÆ° cÃ³ thá»ƒ giá»¯ nguyÃªn.


