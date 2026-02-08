# eval_zeroshot.py

import yaml
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder

from clip.load_clip import load_clip
from clip.clip_forward import get_cls_embedding

from sae.sae_model import SparseAutoencoder

from utils.device import get_device
from utils.logging import eval_logger
from utils.imagenet_utils import load_imagenet_classnames, build_zero_shot_prompts


@torch.no_grad()
def build_text_embeddings(clip_model, tokenizer, text_prompts, device):

    tokens = tokenizer(
        text_prompts,
        padding=True,
        return_tensors="pt"
    ).to(device)

    emb = clip_model.get_text_features(**tokens)
    return F.normalize(emb, dim=-1)



@torch.no_grad()
def evaluate(clip_model, sae, dataloader, text_embs, device, use_sae: bool):
    correct, total = 0, 0

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        img_emb = clip_model.get_image_features(pixel_values=images)

        if i == 0 and total == 0:
            print(f"DEBUG | Input norm BEFORE SAE: {img_emb.norm(dim=-1).mean().item():.2f}")




        if use_sae:
            #ФИКС1
            orig_norm = img_emb.norm(dim=-1, keepdim=True)


            img_emb, z, _ = sae(img_emb)

            #ФИКС2
            img_emb = img_emb * (orig_norm / (img_emb.norm(dim=-1, keepdim=True) + 1e-8))



            if i == 0 and total == 0:
                print(f"DEBUG | Recon norm AFTER SAE:  {img_emb.norm(dim=-1).mean().item():.2f}")
                print(f"DEBUG | Latent max activation: {z.max().item():.4f}")
                print(
                    f"DEBUG | Latent L0 (>0.01):      {(z.abs() > 0.01).float().sum(dim=1).mean().item():.1f} / {z.shape[1]}")

        img_emb = F.normalize(img_emb, dim=-1)

        logits = img_emb @ text_embs.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():
    device = get_device()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    sae_cfg = cfg["sae"]

    clip_model, image_processor, tokenizer = load_clip(cfg["clip_model"])
    clip_model.to(device).eval()


    sae = SparseAutoencoder(
        input_dim=sae_cfg["input_dim"],
        dict_size=sae_cfg["dict_size"],
        l1_alpha=sae_cfg["l1_coeff"],
    )
    sae.load_state_dict(torch.load(cfg["sae_checkpoint"], map_location=device))
    sae.to(device).eval()


    from utils.imagenet_utils import load_imagenet_classnames, build_zero_shot_prompts

    classnames, _, _ = load_imagenet_classnames("data/imagenet_categories.csv")
    text_prompts = build_zero_shot_prompts(classnames)

    datasets = {
        "ImageNet-Sketch": {
            "dataset": ImageFolder(
                root="data/imagenet_subset",
                transform=lambda x: image_processor(
                    images=x, return_tensors="pt"
                )["pixel_values"][0],
            ),
            "classnames": classnames,
            "text_prompts": text_prompts,
        }
    }

    for name, data in datasets.items():
        dataset = data["dataset"]

        loader = DataLoader(
            dataset,
            batch_size=cfg.get("eval_batch_size", 64),
            shuffle=False,
            num_workers=0,
        )

        classnames = (
            data["classnames"]
            if "classnames" in data
            else dataset.classes
        )

        text_embs = build_text_embeddings(
            clip_model, tokenizer, data["text_prompts"], device
        )

        acc_clip = evaluate(
            clip_model, sae, loader, text_embs, device, use_sae=False
        )
        acc_sae = evaluate(
            clip_model, sae, loader, text_embs, device, use_sae=True
        )

        eval_logger.info(
            f"{name} | CLIP: {acc_clip:.4f} | "
            f"CLIP+SAE: {acc_sae:.4f} | "
            f"Delta: {acc_sae - acc_clip:.4f}"
        )


if __name__ == "__main__":
    main()