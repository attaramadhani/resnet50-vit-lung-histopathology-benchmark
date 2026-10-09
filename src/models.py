import re
import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import ViTForImageClassification, ViTConfig

def freeze_vit_layers_regex(model, freeze_layers=[0, 1]):
    """
    Perbaikan Kritis Bug C1:
    Membekukan layer encoder menggunakan regex presisi agar HANYA nomor layer
    yang terdaftar di freeze_layers yang dibekukan, tanpa sengaja membekukan layer 10 dan 11!
    """
    frozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        # Mencocokkan pola: vit.encoder.layer.<angka>. atau encoder.layer.<angka>.
        match = re.search(r"encoder\.layer\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx in freeze_layers:
                param.requires_grad = False
                frozen_params += param.numel()

    trainable_params = total_params - frozen_params
    print(f"[OK] Pembekuan Layer Regex (Layers {freeze_layers}):")
    print(f"  - Parameter Non-Trainable (Beku) : {frozen_params:,}")
    print(f"  - Parameter Trainable (Dilatih) : {trainable_params:,}")
    return frozen_params

def create_vit_model(config, label2id, id2label):
    model_name = config.get("model_name", "WinKawaks/vit-small-patch16-224")
    num_classes = config.get("num_classes", 3)
    img_size = config.get("img_size", 224)

    attn_implementation = config.get("attn_implementation", None)
    config_kwargs = {
        "num_labels": num_classes,
        "image_size": img_size,
        "patch_size": 16,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "label2id": label2id,
        "id2label": id2label,
        "output_hidden_states": False # C11: Matikan hidden states saat training untuk efisiensi
    }
    if attn_implementation:
        config_kwargs["attn_implementation"] = attn_implementation

    vit_config = ViTConfig.from_pretrained(
        model_name,
        **config_kwargs
    )

    model = ViTForImageClassification.from_pretrained(
        model_name,
        config=vit_config,
        ignore_mismatched_sizes=True
    )

    # Perbaiki pembekuan layer dengan regex presisi
    freeze_layers = config.get("freeze_layers", [0, 1])
    if freeze_layers:
        freeze_vit_layers_regex(model, freeze_layers=freeze_layers)

    return model

def create_resnet_model(config, num_classes=3):
    """
    Membangun model ResNet50 menggunakan PyTorch (torchvision) agar setara dan simetris
    dengan pipeline ViT (menyelesaikan Butir C22 di laporan audit).
    """
    weights = tv_models.ResNet50_Weights.DEFAULT
    model = tv_models.resnet50(weights=weights)

    # Opsi pembekuan backbone
    freeze_backbone = config.get("freeze_backbone", False)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("[OK] Backbone ResNet50 dibekukan (Feature Extraction).")
    else:
        print("[OK] Backbone ResNet50 dibuka penuh (Full Fine-Tuning untuk komparasi adil).")

    # Ganti classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model

def build_model(config, label2id, id2label):
    arch = config.get("model_architecture", "vit_s").lower()
    if "vit" in arch:
        return create_vit_model(config, label2id, id2label)
    elif "resnet" in arch:
        num_classes = config.get("num_classes", 3)
        return create_resnet_model(config, num_classes)
    else:
        raise ValueError(f"Arsitektur tidak dikenali: {arch}")
