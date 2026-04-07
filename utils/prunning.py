# utils/pruning.py
import torch
import torch.nn as nn

def extract_pruned_blueprint(teacher_model, prune_ratio=0.5):
    """
    Structured Pruning: L1 norm + BatchNorm γ
    Trả về channel_config cho Student (blueprint)
    """
    blueprint = []
    for name, module in teacher_model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 16:
            # Ưu tiên BatchNorm γ nếu có
            importance = None
            bn_name = name.replace(".conv", ".bn") if ".conv" in name else name + ".bn"
            for n, m in teacher_model.named_modules():
                if bn_name in n and isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    importance = m.weight.data.abs()
                    break
            if importance is None:
                importance = module.weight.data.abs().sum(dim=(1, 2, 3))  # L1 norm

            keep_channels = int(importance.numel() * (1 - prune_ratio))
            keep_channels = max(keep_channels, 16)
            blueprint.append(keep_channels)

    # Lấy 5 giá trị cho encoder (stem + 4 down)
    config = blueprint[:5]
    print(f"✅ Pruned blueprint: {config} (prune_ratio={prune_ratio})")
    return tuple(config)