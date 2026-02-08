import torch

import torch.nn.functional as F


@torch.no_grad()
def get_cls_embedding(model, images):

    img_emb = model.get_image_features(pixel_values=images)
    img_emb = F.normalize(img_emb, dim=-1)
    return img_emb
