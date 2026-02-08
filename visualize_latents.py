import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder

LATENTS_PATH = "artifacts/activations/latents.npy"
CSV_PATH = "artifacts/interpretations/sae_latents.csv"
DATA_ROOT = "data/imagenet_subset"

TOP_K = 9
TOP_LATENTS = 20


latents = np.load(LATENTS_PATH)
dataset = ImageFolder(root=DATA_ROOT, transform=None)
assert len(dataset) == latents.shape[0], f"Size mismatch!"

interpretations = {}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interpretations[int(row["latent_id"])] = row["interpretation"]
else:
    interpretations = {i: "No interpretation" for i in range(latents.shape[1])}

print(" Selecting MOST interpretable latents (strength + specificity)...")

max_acts = latents.max(axis=0)  # [dict_size]
active_counts = (latents > 0.1).sum(axis=0)  # значимые активации (>0.1)

min_active = 40  # минимум активаций для статистической значимости
max_active = 3000  # максимум для монозначности (при 50к изображениях)

valid_mask = (active_counts >= min_active) & (active_counts <= max_active)
valid_indices = np.where(valid_mask)[0]

if len(valid_indices) == 0:

    print("⚠️  No latents in ideal range (40-3000 activations). Expanding search...")
    valid_mask = active_counts >= 20
    valid_indices = np.where(valid_mask)[0]

interpretability = np.zeros_like(max_acts)
interpretability[valid_indices] = max_acts[valid_indices] / np.log1p(active_counts[valid_indices])

top_indices = np.argsort(interpretability)[-TOP_LATENTS:][::-1]
LATENT_IDS = top_indices.tolist()

print(f" Selected {len(LATENT_IDS)} highly interpretable latents:")
print(f"{'Rank':<6} {'Latent ID':<12} {'Max Act':<12} {'Active Imgs':<15} {'Interp Score':<15}")
print("-" * 70)
for i, idx in enumerate(LATENT_IDS):
    score = interpretability[idx]
    print(f"{i + 1:<6} {idx:<12} {max_acts[idx]:<12.3f} {active_counts[idx]:<15} {score:<15.4f}")

os.makedirs("artifacts/visualizations", exist_ok=True)
successful = 0

for rank, k in enumerate(LATENT_IDS, 1):
    acts = latents[:, k]
    active_mask = acts > 0.1

    if active_mask.sum() < TOP_K:
        continue

    top_idx = np.argsort(acts[active_mask])[-TOP_K:][::-1]
    orig_idx = np.where(active_mask)[0][top_idx]

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    title = (f"Latent #{rank} (ID: {k}) | Max: {max_acts[k]:.2f} | "
             f"Active: {active_counts[k]}\n{interpretations.get(k, 'No interpretation')}")
    fig.suptitle(title, fontsize=9, y=0.98)

    for ax, idx in zip(axes.flatten(), orig_idx):
        img, _ = dataset[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.text(5, 5, f"{acts[idx]:.2f}", color='white', fontsize=7,
                bbox=dict(facecolor='black', alpha=0.6, pad=1))

    plt.tight_layout()
    plt.savefig(f"artifacts/visualizations/latent_{k:04d}.png", dpi=200, bbox_inches='tight')
    plt.close()
    successful += 1

print(f" Saved {successful} high-quality visualizations to artifacts/visualizations/")
print(" Tips for analysis:")
print("  Open latent_XXXX.png files to see visual patterns")
print("  Look for common visual features across the 9 images")
print("  Best latents will show consistent patterns (e.g., all diagonal lines)")
print("  Ignore latents showing random objects — they're not monosemantic")