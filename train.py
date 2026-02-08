# train.py
import os
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from clip.load_clip import load_clip
from sae.sae_model import SparseAutoencoder
from sae.metrics import compute_l0, explained_variance_ratio

from utils.logging import train_logger
from utils.device import get_device
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    cfg = load_config("configs/config.yaml")

    set_seed(cfg["seed"])
    device = get_device()


    data_root = cfg["data_root"]
    ckpt_path = cfg["sae_checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)


    clip_model, image_processor, tokenizer = load_clip(
        model_name=cfg["clip_model"]
    )
    clip_model.to(device).eval()

    for p in clip_model.parameters():
        p.requires_grad = False


    def clip_transform(img):
        return image_processor(
            images=img,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)  # [3, 224, 224]


    dataset = ImageFolder(
        root=data_root,
        transform=clip_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )


    sae = SparseAutoencoder(
        input_dim=cfg["sae"]["input_dim"],   # 512 for ViT-B/32
        dict_size=cfg["sae"]["dict_size"],   # e.g. 8192
        l1_alpha=cfg["sae"]["l1_coeff"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=cfg["lr"]
    )

    for epoch in range(cfg["epochs"]):
        sae.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for step, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)  # [B, 3, 224, 224]

            with torch.no_grad():
                targets = clip_model.get_image_features(pixel_values=images)


            recon, z, l1_loss = sae(targets)
            mse_loss = F.mse_loss(recon, targets)
            loss = mse_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

            optimizer.step()

            with torch.no_grad():
                sae.decoder.weight.data[:] = F.normalize(
                    sae.decoder.weight.data, dim=0
                )

            if step % 100 == 0:
                col_norms = sae.decoder.weight.data.norm(dim=0)
                print(
                    f"Decoder column norms | min: {col_norms.min().item():.4f} | max: {col_norms.max().item():.4f} | mean: {col_norms.mean().item():.4f}")

            l0 = compute_l0(z, threshold=1e-3)
            r2 = explained_variance_ratio(recon, targets)

            if step % cfg["log_interval"] == 0:
                train_logger.info(
                    f"Epoch {epoch:03d} | Step {step:05d} | "
                    f"Loss {loss.item():.5f} | "
                    f"MSE {mse_loss.item():.5f} | "
                    f"L1 {l1_loss.item():.5f} | "
                    f"L0 {l0:.2f} | "
                    f"R2 {r2:.4f}"
                )

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(
                sae.state_dict(),
                f"artifacts/checkpoints/sae_epoch_{epoch + 1}.pt",
            )

    torch.save(sae.state_dict(), "artifacts/checkpoints/sae_final.pt")


if __name__ == "__main__":
    main()