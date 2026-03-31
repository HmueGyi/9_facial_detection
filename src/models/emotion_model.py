import torch
import torch.nn as nn
from torchvision import models


class EmotionModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EmotionModel, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError("Checkpoint must be a state_dict dictionary or contain one under 'model_state_dict'/'state_dict'.")


def _strip_optional_prefix(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _add_backbone_prefix_if_needed(state_dict):
    has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict)
    has_raw_backbone_keys = any(k.startswith("features.") or k.startswith("classifier.") for k in state_dict)
    if has_backbone_prefix or not has_raw_backbone_keys:
        return state_dict
    return {f"backbone.{k}": v for k, v in state_dict.items()}


def _load_with_key_adapters(model, state_dict):
    variants = [
        ("as-is", state_dict),
        ("strip 'module.'", _strip_optional_prefix(state_dict, "module.")),
    ]

    variants.extend([
        ("add 'backbone.'", _add_backbone_prefix_if_needed(variants[0][1])),
        ("strip 'module.' + add 'backbone.'", _add_backbone_prefix_if_needed(variants[1][1])),
    ])

    seen = set()
    unique_variants = []
    for name, variant in variants:
        signature = tuple(variant.keys())
        if signature in seen:
            continue
        seen.add(signature)
        unique_variants.append((name, variant))

    last_error = None
    for _, variant in unique_variants:
        try:
            model.load_state_dict(variant)
            return
        except RuntimeError as exc:
            last_error = exc

    tried = ", ".join(name for name, _ in unique_variants)
    raise RuntimeError(
        f"Unable to load checkpoint after trying key adapters [{tried}]. "
        f"Last error: {last_error}"
    )


def get_model(model_path=None, device='cpu', num_classes=6):
    model = EmotionModel(num_classes=num_classes, pretrained=False)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        _load_with_key_adapters(model, state_dict)
            
    model.to(device)
    model.eval()
    return model
