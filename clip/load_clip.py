

from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
import torch


def load_clip(model_name):
    model = CLIPModel.from_pretrained(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    return model, image_processor, tokenizer