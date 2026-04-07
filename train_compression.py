# train_compression.py
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from dataloaders.dataset import build_dataset, Normalize, RandomGenerator, ToTensor
from networks.net_factory import net_factory
from networks.gated_unet import GatedUNet
from utils.pruning import extract_pruned_blueprint
from utils.compression_loss import CompressionLoss
from utils.evaluation import evaluate_segmentation_dataset, build_evaluation_output_dir, save_evaluation_artifacts
from utils.losses import DiceLoss

PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train Student with Pruning + Gating + Distillation")
parser.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir", "cvc"])
parser.add_argument("--root_path", type=str, default="data/Kvasir-SEG")
parser.add_argument("--teacher_model", type=str, default="unet_resnet152", help="Teacher model đã train")
parser.add_argument("--teacher_exp", type=str, default="supervised", help="Experiment name của Teacher")
parser.add_argument("--prune_ratio", type=float, default=0.5, help="Tỷ lệ pruning (0.4 - 0.6)")
parser.add_argument("--lambda_distill", type=float, default=0.3)
parser.add_argument("--lambda_sparsity", type=float, default=0.3)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--exp", type=str, default="student_compression")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--deterministic", type=int, default=1)

args = parser.parse_args()

# ====================== SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(args.gpu))

# 1. Load Teacher (đã train sẵn)
snapshot_path = PROJECT_ROOT / "logs/model/supervised" / args.teacher_exp
teacher = net_factory(net_type=args.teacher_model, in_chns=3, class_num=2, mode="test").to(device)

# Load best checkpoint của Teacher
checkpoint_path = snapshot_path / "weights" / f"{args.dataset}_{args.teacher_model}_best.pth"
if not checkpoint_path.exists():
    checkpoint_path = list((snapshot_path / "weights").glob("*best*.pth"))[0]

teacher.load_state_dict(torch.load(checkpoint_path, map_location=device))
teacher.eval()
print(f"=> Loaded Teacher from: {checkpoint_path}")

# 2. Pruning → lấy blueprint
blueprint = extract_pruned_blueprint(teacher, prune_ratio=args.prune_ratio)

# 3. Tạo Student
student = GatedUNet(in_channels=3, num_classes=2, 
                   channel_config=blueprint).to(device)
print(f"=> Created Student with blueprint: {blueprint}")

# 4. Data
train_transform = transforms.Compose([Normalize(), RandomGenerator([352, 352])])
eval_transform = transforms.Compose([Normalize(), ToTensor()])

db_train = build_dataset(args.dataset, args.root_path, split="train", transform=train_transform)
db_val   = build_dataset(args.dataset, args.root_path, split="val",   transform=eval_transform)

trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader   = DataLoader(db_val,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# 5. Loss & Optimizer
criterion = CompressionLoss(num_classes=2, 
                           lambda_distill=args.lambda_distill,
                           lambda_sparsity=args.lambda_sparsity)
optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)

# ====================== TRAINING ======================
best_dice = 0.0
save_dir = PROJECT_ROOT / "logs/model/compression" / args.exp
save_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, args.max_epochs + 1):
    student.train()
    epoch_loss = 0.0

    for batch in trainloader:
        image = batch["image"].to(device)
        target = batch["label"].to(device)

        with torch.no_grad():
            teacher_logits = teacher(image)

        student_logits = student(image)
        gates = student.get_all_gates()

        loss, L_seg, L_distill, L_sparsity = criterion(student_logits, teacher_logits, gates, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    if epoch % 5 == 0 or epoch == args.max_epochs:
        student.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in valloader:
                image = batch["image"].to(device)
                target = batch["label"].to(device)
                pred = student(image)
                # Tính Dice đơn giản
                pred = torch.argmax(pred, dim=1)
                dice = (2 * (pred * target).sum() + 1e-8) / (pred.sum() + target.sum() + 1e-8)
                val_dice += dice.item()

        val_dice /= len(valloader)
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss/len(trainloader):.5f} | Val Dice: {val_dice:.5f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(student.state_dict(), save_dir / "best_student.pth")
            print(f"   → Saved best Student (Dice = {best_dice:.5f})")

print(f"\n TRAINING STUDENT MODEL hoàn tất! Best Dice = {best_dice:.5f}")
print(f"   Model saved at: {save_dir}/best_student.pth")