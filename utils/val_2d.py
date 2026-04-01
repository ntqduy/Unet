import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric


def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0, 0

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95


def _extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        tensor_candidates = [item for item in model_output if torch.is_tensor(item) and item.ndim >= 4]
        if not tensor_candidates:
            raise ValueError("Model output does not contain segmentation logits.")
        return tensor_candidates[0]
    return model_output


def predict_single_image(image, model, patch_size=(256, 256), device=None, classes=2):
    if device is None:
        device = next(model.parameters()).device

    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.float()
    original_size = tuple(image.shape[-2:])

    resized_image = F.interpolate(image, size=patch_size, mode="bilinear", align_corners=False)

    model.eval()
    with torch.no_grad():
        logits = _extract_logits(model(resized_image.to(device)))
        if logits.shape[1] == 1 or classes == 1:
            prediction = (torch.sigmoid(logits) > 0.5).long()
        else:
            prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)

    prediction = F.interpolate(prediction.float(), size=original_size, mode="nearest").squeeze(1).long().cpu()
    return prediction


def test_single_volume(image, label, model, classes, patch_size=(256, 256), device=None, return_prediction=False):
    if label.ndim == 2:
        label = label.unsqueeze(0)

    label = label.long()
    prediction = predict_single_image(image=image, model=model, patch_size=patch_size, device=device, classes=classes)

    prediction_np = prediction.numpy()
    label_np = label.cpu().numpy()

    metric_list = []
    for class_index in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction_np[0] == class_index, label_np[0] == class_index))
    if return_prediction:
        return metric_list, prediction
    return metric_list
