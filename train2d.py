import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import Normalize, RandomGenerator, ToTensor, build_dataset, list_available_datasets
from networks.net_factory import list_models, net_factory
from utils import losses, val_2d
from utils.visualization import save_triplet_visualization


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        tensor_candidates = [item for item in model_output if torch.is_tensor(item) and item.ndim >= 4]
        if not tensor_candidates:
            raise ValueError("Model output does not contain segmentation logits.")
        return tensor_candidates[0]
    return model_output


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT), help="dataset root path")
parser.add_argument("--dataset", type=str, default="kvasir", choices=list_available_datasets(), help="dataset name")
parser.add_argument("--exp", type=str, default="supervised", help="experiment name")
parser.add_argument("--model", type=str, default="unet", choices=list_models(), help="model name")
parser.add_argument("--train_split", type=str, default="train", choices=["train", "val", "test"], help="training split")
parser.add_argument("--val_split", type=str, default="val", choices=["train", "val", "test"], help="evaluation split")
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--eval_interval", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3, help="number of image channels to load")
parser.add_argument("--encoder_pretrained", type=int, default=0, help="only used by unet_resnet152")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_num_samples", type=int, default=5)

args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=args.num_classes)


def save_validation_visualizations(vis_samples, output_dir):
    output_dir = Path(output_dir)
    for sample in vis_samples:
        save_triplet_visualization(
            image=sample["image"],
            label=sample["label"],
            prediction=sample["prediction"],
            output_dir=output_dir,
            case_name=sample["case"],
        )


def train(args, snapshot_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_mode = "grayscale" if args.in_channels == 1 else "rgb"

    train_transform = transforms.Compose([Normalize(), RandomGenerator(args.patch_size)])
    eval_transform = transforms.Compose([Normalize(), ToTensor()])

    db_train = build_dataset(
        dataset_name=args.dataset,
        base_dir=args.root_path,
        split=args.train_split,
        transform=train_transform,
        image_mode=image_mode,
    )
    db_val = build_dataset(
        dataset_name=args.dataset,
        base_dir=args.root_path,
        split=args.val_split,
        transform=eval_transform,
        image_mode=image_mode,
    )

    model_kwargs = {"mode": "train"}
    if args.model == "unetr":
        model_kwargs["image_size"] = tuple(args.patch_size)
    if args.model == "unet_resnet152":
        model_kwargs["encoder_pretrained"] = bool(args.encoder_pretrained)
    model = net_factory(
        net_type=args.model,
        in_chns=db_train.in_channels,
        class_num=args.num_classes,
        **model_kwargs,
    ).to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=device.type == "cuda",
    )

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = CrossEntropyLoss()

    logging.info("Start training")
    logging.info("%d iterations per epoch", len(trainloader))
    logging.info("Train split: %s | Val split: %s", args.train_split, args.val_split)

    model.train()
    iter_num = 0
    best_performance = 0.0
    max_epoch = args.max_iterations // max(len(trainloader), 1) + 1

    for _ in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            image_batch = sampled_batch["image"].to(device)
            label_batch = sampled_batch["label"].to(device)

            logits = extract_logits(model(image_batch))
            loss_ce = ce_loss(logits, label_batch.long())
            outputs_soft = torch.softmax(logits, dim=1)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            logging.info("iteration %d : loss : %f", iter_num, loss.item())

            if iter_num % args.eval_interval == 0:
                model.eval()
                metric_list = []
                vis_samples = []

                with torch.no_grad():
                    for sampled_val in valloader:
                        need_prediction = bool(args.save_visualizations) and len(vis_samples) < args.vis_num_samples
                        metric_output = val_2d.test_single_volume(
                            sampled_val["image"],
                            sampled_val["label"],
                            model,
                            classes=args.num_classes,
                            patch_size=args.patch_size,
                            device=device,
                            return_prediction=need_prediction,
                        )

                        if need_prediction:
                            metric_i, prediction = metric_output
                            case_name = sampled_val["case"][0] if isinstance(sampled_val["case"], (list, tuple)) else str(sampled_val["case"])
                            vis_samples.append(
                                {
                                    "case": case_name,
                                    "image": sampled_val["image"][0],
                                    "label": sampled_val["label"][0],
                                    "prediction": prediction[0],
                                }
                            )
                        else:
                            metric_i = metric_output

                        metric_list.append(np.array(metric_i))

                metric_array = np.stack(metric_list, axis=0).mean(axis=0)
                performance = float(np.mean(metric_array[:, 0]))

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = Path(snapshot_path) / f"iter_{iter_num}_dice_{round(best_performance, 4)}.pth"
                    save_best_path = Path(snapshot_path) / f"{args.model}_best_model.pth"
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to %s", save_mode_path)

                    if args.save_visualizations and vis_samples:
                        vis_dir = Path(snapshot_path) / "visualizations" / f"iter_{iter_num}"
                        save_validation_visualizations(vis_samples, vis_dir)

                logging.info("iteration %d : mean_dice : %f", iter_num, performance)
                model.train()

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            break


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    snapshot_path = PROJECT_ROOT / "logs" / "model" / "supervised" / args.exp
    snapshot_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(Path(__file__).resolve(), snapshot_path / "train2d.py")

    logging.basicConfig(
        filename=str(snapshot_path / "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, str(snapshot_path))
