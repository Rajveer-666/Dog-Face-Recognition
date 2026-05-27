# =============================================================
# app/services/breed_service.py  --  Breed classification branch
# =============================================================
# Input  : PIL RGB image (full upload, not cropped)
# Output : (breed_label: str, confidence: float)
# =============================================================

from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image

from app.core.config import settings

# ------------------------------------------------------------------
# Preprocessing transform
# Matches training transform from breed notebook:
#   IMG_SIZE = 320, ImageNet mean/std normalisation
# ------------------------------------------------------------------
_breed_transform = transforms.Compose([
    transforms.Resize((settings.BREED_IMG_SIZE, settings.BREED_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_breed(pil_img: Image.Image, registry) -> Tuple[str, float]:
    """
    Run breed classification on the full input image.

    Args:
        pil_img  : PIL RGB image (the raw upload, not cropped)
        registry : ModelRegistry instance holding breed_model + idx_to_breed

    Returns:
        breed_label : human-readable breed string, e.g. 'Golden_Retriever'
        confidence  : softmax probability of top prediction (0-1)
    """
    # Preprocess
    x = _breed_transform(pil_img).unsqueeze(0).to(settings.DEVICE)

    # Inference
    with torch.no_grad():
        logits = registry.breed_model(x)          # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]  # (num_classes,)
        idx    = int(torch.argmax(probs).item())
        conf   = float(probs[idx].item())

    # Map index -> readable breed name
    raw_label   = registry.idx_to_breed.get(idx, "unknown")
    # Strip Stanford-Dogs folder prefix like 'n02085620-'
    breed_label = raw_label.split("-", 1)[-1].replace("_", " ")

    return breed_label, conf
