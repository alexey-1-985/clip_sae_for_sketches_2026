# auto_interpret.py
# Auto-interpretation of SAE latents using VLM (OpenRouter)

import os
import csv
import yaml
import json
import torch
import numpy as np
import requests
from tqdm import tqdm

from utils.device import get_device

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def load_latents(path):
    """
    latents.npy: shape [N_images, dict_size]
    """
    latents = np.load(path)
    assert latents.ndim == 2
    return latents


def load_image_captions(path):
    """
    captions.json:
    {
        "0": "a photo of a dog running on grass",
        "1": "a truck on a road",
        ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_openrouter(prompt, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in interpreting visual features learned by neural networks. "
                    "Given several image descriptions that strongly activate a latent feature, "
                    "describe the shared visual concept in ONE concise sentence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 50,
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def build_prompt(latent_idx, captions):
    examples = "\n".join(f"- {c}" for c in captions)

    return (
        f"Latent feature #{latent_idx} is strongly activated by the following images:\n"
        f"{examples}\n\n"
        "Describe the shared visual concept in one sentence."
    )



def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    os.makedirs("artifacts/interpretations", exist_ok=True)


    latents = load_latents("artifacts/activations/latents.npy")
    captions = load_image_captions("artifacts/activations/captions.json")

    num_images, dict_size = latents.shape

    n_interpret = min(8000, dict_size)
    top_k = cfg.get("interpret_top_k", 8)


    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model_name = cfg.get("openrouter_model", "openai/gpt-3.5-turbo")

    output_csv = "artifacts/interpretations/sae_latents.csv"


    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["latent_id", "interpretation"])

        for latent_idx in tqdm(range(n_interpret), desc="Interpreting SAE latents"):
            acts = latents[:, latent_idx]
            top_indices = np.argsort(acts)[-top_k:][::-1]

            top_captions = [captions[str(i)] for i in top_indices]

            prompt = build_prompt(latent_idx, top_captions)

            try:
                interpretation = query_openrouter(
                    prompt=prompt,
                    api_key=api_key,
                    model=model_name,
                )
            except Exception as e:
                interpretation = f"ERROR: {e}"

            writer.writerow([latent_idx, interpretation])


if __name__ == "__main__":
    main()