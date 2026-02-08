import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder

from torchvision.datasets import CIFAR10
from torchvision import transforms


LATENTS_PATH = "artifacts/activations/latents.npy"
CSV_PATH = "artifacts/interpretations/sae_latents.csv"
DATA_ROOT = "data/imagenet_subset"

TOP_K = 9
LATENT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,30, 31, 32, 33, 34, 35,36, 37, 38, 39,40, 41, 42, 43, 44, 45,46, 47, 48, 49,50, 51, 52, 53, 54, 55,56, 57, 58, 59, 50, 51, 52, 53, 54, 55,56, 57, 58, 59,60, 61, 62, 63, 64, 65,66, 67, 68, 69,70, 71, 72, 73, 74, 75,76, 77, 78, 79, 80, 298, 299]  # какие латенты визуализировать

latents = np.load(LATENTS_PATH)  # [N_images, dict_size]

dataset = ImageFolder(
    root=DATA_ROOT,
    transform=None
)

assert len(dataset) == latents.shape[0], \
    "Dataset size ≠ number of latent vectors!"

interpretations = {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        interpretations[int(row["latent_id"])] = row["interpretation"]


os.makedirs("artifacts/visualizations", exist_ok=True)

for k in LATENT_IDS:
    acts = latents[:, k]

    active_idx = np.where(acts > 1e-3)[0]
    if len(active_idx) == 0:
        print(f"Latent {k}: no active images")
        continue

    top_idx = active_idx[np.argsort(acts[active_idx])[-TOP_K:]][::-1]

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.suptitle(
        f"Latent {k}\n{interpretations.get(k, 'No interpretation')}",
        fontsize=10
    )

    for ax, idx in zip(axes.flatten(), top_idx):
        img, _ = dataset[idx]
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"artifacts/visualizations/latent_{k}.png", dpi=200)
    plt.close()

print("✅ Visualizations saved to artifacts/visualizations/")