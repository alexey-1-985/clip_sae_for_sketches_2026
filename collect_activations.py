import os
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from clip.load_clip import load_clip
from sae.sae_model import SparseAutoencoder
from utils.device import get_device


def load_imagenet_classnames(csv_path: str):
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("synset").reset_index(drop=True)
    classnames = [words.split(",")[0].strip() for words in df["words"]]
    csv_synsets = df["synset"].tolist()
    return classnames, csv_synsets


@torch.no_grad()
def main():
    device = get_device()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    sae_cfg = cfg["sae"]
    os.makedirs("artifacts/activations", exist_ok=True)

    # Load CLIP
    clip_model, image_processor, tokenizer = load_clip(cfg["clip_model"])
    clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Load SAE
    sae = SparseAutoencoder(
        input_dim=sae_cfg["input_dim"],
        dict_size=sae_cfg["dict_size"],
        l1_alpha=sae_cfg["l1_coeff"],
    )
    try:
        sae.load_state_dict(torch.load(cfg["sae_checkpoint"], map_location=device, weights_only=True))
    except TypeError:
        sae.load_state_dict(torch.load(cfg["sae_checkpoint"], map_location=device))
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad = False

    # Load classnames
    csv_path = cfg.get("imagenet_csv", "data/imagenet_categories.csv")
    classnames, csv_synsets = load_imagenet_classnames(csv_path)

    # Dataset
    data_root = cfg.get("eval_data_root", "data/imagenet_subset")
    dataset = ImageFolder(
        root=data_root,
        transform=lambda x: image_processor(
            images=x, return_tensors="pt"
        )["pixel_values"][0],
    )

    # Sanity check class order
    assert len(dataset.classes) == len(csv_synsets), "Class count mismatch"
    min_check = min(10, len(dataset.classes))
    for i in range(min_check):
        assert dataset.classes[i] == csv_synsets[i], f"Class order mismatch at index {i}"
    print(f"✅ Class order validated ({min_check} samples)")

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------------------------------
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: правильный сбор латентов
    # -------------------------------------------------
    all_latents = []
    captions = {}
    idx = 0

    for images, labels in tqdm(loader, desc="Collecting SAE activations"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1. НЕНОРМАЛИЗОВАННЫЕ эмбеддинги (как при обучении!)
        z_img = clip_model.get_image_features(pixel_values=images)  # ← КРИТИЧЕСКИ ВАЖНО!

        # 2. Получить ЛАТЕНТЫ (не реконструкцию!)
        # Вариант A (рекомендуется): напрямую через encode
        sparse_latents = sae.encode(z_img)  # shape: [B, dict_size]

        # Вариант B (альтернатива): через forward
        # _, sparse_latents, _ = sae(z_img)  # второй элемент = латенты

        all_latents.append(sparse_latents.cpu().numpy())

        # 3. Сохранить описания
        for lbl in labels.cpu().tolist():
            class_name = classnames[lbl]
            caption = f"a sketch of a {class_name}"
            captions[str(idx)] = caption
            idx += 1

    # Validation & save
    if not all_latents:
        raise RuntimeError("No activations collected!")

    latents = np.concatenate(all_latents, axis=0)  # shape: [N, dict_size]

    assert latents.shape[0] == len(captions) == idx, "Size mismatch"

    # Диагностика качества
    active_per_image = (latents > 1e-3).sum(axis=1).mean()
    max_activation = latents.max()
    print(f"📊 Avg active latents per image: {active_per_image:.1f} / {latents.shape[1]}")
    print(f"📊 Max latent activation: {max_activation:.4f}")
    if max_activation < 0.5:
        print("⚠️  Warning: max activation too low (<0.5) — check SAE training quality")
    if active_per_image > latents.shape[1] * 0.1:
        print("⚠️  Warning: SAE may not be sparse enough (>10% active)")

    # Save
    np.save("artifacts/activations/latents.npy", latents)
    with open("artifacts/activations/captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)

    print(f"\n✅ Successfully saved {latents.shape[0]} samples:")
    print(f"   - artifacts/activations/latents.npy (shape: {latents.shape})")
    print(f"   - artifacts/activations/captions.json ({len(captions)} captions)")


if __name__ == "__main__":
    main()