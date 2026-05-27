# =============================================================
# app/services/identity_service.py  --  Embedding + gallery match
# =============================================================
# extract_embedding : face crop -> L2-normalised feature vector
# match_identity    : query vector -> (dog_id, cosine_score)
# =============================================================

from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from app.core.config import settings


# ------------------------------------------------------------------
# Preprocessing for the identity encoder
# Uses ImageNet mean/std at ENCODER_IMG_SIZE (default 224)
# ------------------------------------------------------------------
_encoder_transform = transforms.Compose([
    transforms.Resize((settings.ENCODER_IMG_SIZE, settings.ENCODER_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_embedding(
    face_bgr: np.ndarray,
    registry,
) -> np.ndarray:
    """
    Convert a face crop (BGR numpy array) into an L2-normalised
    embedding vector using the loaded identity encoder.

    Args:
        face_bgr : OpenCV BGR crop from detect_and_crop_face()
        registry : ModelRegistry holding encoder

    Returns:
        embedding : float32 numpy array of shape (D,), L2-normalised
    """
    # BGR -> RGB -> PIL
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(face_rgb)

    x = _encoder_transform(pil_face).unsqueeze(0).to(settings.DEVICE)

    with torch.no_grad():
        emb = registry.encoder(x)          # (1, D)
        # Handle models that return tuple (embedding, logits)
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)

    # L2-normalise query embedding
    norm = np.linalg.norm(emb) + 1e-12
    return emb / norm


def match_identity(
    query_embedding: np.ndarray,
    registry,
) -> Tuple[str, float]:
    """
    Compare query embedding against the pre-loaded gallery using
    cosine similarity (dot product of L2-normalised vectors).

    Args:
        query_embedding : L2-normalised float32 array (D,)
        registry        : ModelRegistry holding gallery_embs + gallery_ids

    Returns:
        dog_id      : identity string from idx2class, or 'unknown'
        match_score : cosine similarity of best match (0-1)
    """
    # gallery_embs is already L2-normalised at load time
    # shape: (N, D)  x  (D,) -> (N,)  cosine similarities
    sims     = registry.gallery_embs @ query_embedding
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < settings.MATCH_THRESHOLD:
        return "unknown", best_score

    return registry.gallery_ids[best_idx], best_score
