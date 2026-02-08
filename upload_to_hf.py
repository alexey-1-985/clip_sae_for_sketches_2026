from huggingface_hub import HfApi
import os

repo_id = "Aleksey110/clip-sae-sketch"
api = HfApi()

api.create_repo(repo_id=repo_id, exist_ok=True)

api.upload_file(
    path_or_fileobj="artifacts/checkpoints/sae_final.pt",
    path_in_repo="sae_weights.pt",
    repo_id=repo_id
)
api.upload_file(
    path_or_fileobj="configs/config.yaml",
    path_in_repo="config.yaml",
    repo_id=repo_id
)
print(f"Uploaded to https://huggingface.co/{repo_id}")