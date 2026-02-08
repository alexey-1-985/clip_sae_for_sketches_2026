# CLIP Sparse Autoencoder (SAE) Project

This project extracts **monosemantic visual features** from CLIP embeddings of *sketches* (ImageNet-Sketch) using Sparse Autoencoders (SAE). Unlike real photos, sketches contain only contours and shapes — making them an ideal domain to study *fundamental visual primitives* (lines, curves, textures) learned by vision models.


## Why Sketches?

| Domain | Visual Information | What SAEs Reveal |
|--------|-------------------|------------------|
|**Real photos** | Colors, textures, lighting, details | Semantic concepts ("dog", "car") + noisy features |
| **Sketches**  | **Only contours and shapes** |  **Pure geometric primitives** (lines, curves, angles) |

Sketches act as a **magnifying glass** for low-level vision:
- edges
- curves
- junctions
- repetitive strokes

This allows SAE latents to become **monosemantic**, each representing a single geometric concept.


# Optional: init submodule for reference code
git submodule add https://github.com/saprmarks/dictionary_learning.git third_party/dictionary_learning

Interpretation Pipeline
python train.py                # SAE training on CLIP embeddings
python eval_zeroshot.py        # Evaluate classification impact (Δ ≈ -5–10%)
python collect_activations.py  # Collect sparse activations
python auto_interpret.py       # LLM-based interpretation (requires OPENROUTER_API_KEY)
python visualize_latents.py    # Visual verification of top latents


clip_sae_project/
│
├── README.md
├── requirements.txt
├── .gitignore
│── collect_activations.py      # Activation collection for interpretation
├── train.py                    # SAE training on CLIP embeddings
├── eval_zeroshot.py            # Zero-shot classification evaluation
├── auto_interpret.py           # LLM-based latent interpretation
├── upload_to_hf.py                  
├── visualize_latents.py        # Visual verification of latent concepts
├── visualize_latents_manual_selection.py   # Visual verification of latent concepts manual
│
├── clip/
│   ├── __init__.py
│   ├── load_clip.py                 
│   └── clip_forward.py              
│
├── sae/
│   ├── __init__.py
│   ├── sae_model.py
│   └── metrics.py
│
├── utils/
│   ├── logging.py                   
│   ├── seed.py                      
│   └── device.py
│
├── configs/
│   └── config.yaml         # Hyperparameters (l1_coeff, dict_size, etc.)
│
├── data/
│   └── imagenet_subset/    # ImageNet-Sketch dataset (50k images)
│
├── artifacts/                       
│   ├── checkpoints/        # SAE weights (sae_final.pt)
│   │   └── sae_epoch_XX.pt
│   ├── activations/        # Latent vectors (latents.npy, captions.json)
│   │   └── latents.npy
│   ├── interpretations/    # LLM interpretations (sae_latents.csv)
│   │   └── sae_latents.csv
│   └── visualizations/     # Latent visualizations (latent_XXXX.png)
│
├── logs/                   # Training and evaluation logs
│   ├── train.log
│   └── eval.log
│
└── notebooks/
    └── kandinsky_steering.ipynb



data/
└── imagenet_subset/
    ├── n01440764/
    │   ├── xxx.jpg
    │   ├── yyy.jpg
    │   └── ...
    ├── n01443537/
    ├── n01484850/
    └── ...

Metrics

The metrics dataset CIFAR-10 was also used as an example. However, the small image sizes are of no interest.

 CIFAR-10 | CLIP: 0.8880 | CLIP+SAE: 0.8882 | Delta: 0.0002
 ImageNet-Sketch | CLIP: 0.4012 | CLIP+SAE: 0.3250 | Delta: -0.0762


 CIFAR-10 |Epoch 009 | Step 00700 | Loss 0.00001 | MSE 0.00000 | L1 0.00000 | L0 1619.67 | R2 0.9941
 ImageNet-Sketch | Epoch 004 | Step 00700 | Loss 0.09708 | MSE 0.03655 | L1 0.06053 | L0 473.55 | R2 0.6003