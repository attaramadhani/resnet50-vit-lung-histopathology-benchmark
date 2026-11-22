"""Verify weight equivalence between timm ViT-S/16 and HuggingFace WinKawaks/vit-small-patch16-224."""
import json
import torch
import timm
from transformers import ViTForImageClassification


def main():
    torch.manual_seed(42)
    x = torch.randn(4, 3, 224, 224)

    m_timm = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval()
    m_hf = ViTForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224").eval()

    with torch.no_grad():
        y_timm = m_timm(x)
        y_hf = m_hf(pixel_values=x).logits

    max_diff = float((y_timm - y_hf).abs().max().item())
    top1_agree = float((y_timm.argmax(1) == y_hf.argmax(1)).float().mean().item())
    out = {
        "timm_model": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "hf_model": "WinKawaks/vit-small-patch16-224",
        "max_abs_diff": max_diff,
        "top1_agreement": top1_agree,
        "identical": max_diff < 1e-3 and top1_agree == 1.0,
    }
    with open("vit_checkpoint_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
