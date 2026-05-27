# =============================================================
# app/services/model_registry.py  --  Load all models once
# =============================================================

import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from app.core.config import settings


# ------------------------------------------------------------------
# Breed classifier architecture
# Matches: ConvNeXt-Tiny + custom head used in breed notebook
# ------------------------------------------------------------------
def _build_breed_model(num_classes: int) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# ------------------------------------------------------------------
# Identity encoder architecture
# IMPORTANT: If your encoder uses a different backbone (e.g. ResNet50,
# EfficientNet, ViT), replace the body of this function to match
# your dog_encoder_best.pth architecture exactly.
# ------------------------------------------------------------------
def _build_encoder(embedding_dim: int) -> nn.Module:
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, embedding_dim)
    return backbone


class ModelRegistry:
    """Holds references to all loaded models and gallery data."""

    def __init__(self):
        self.breed_model   = None
        self.idx_to_breed  = None   # dict: int -> breed label string
        self.face_detector = None
        self.encoder       = None
        self.gallery_embs  = None   # np.ndarray (N, D), L2-normalised
        self.gallery_ids   = None   # list of dog identity strings length N

    # ------------------------------------------------------------------
    def load_all(self):
        print("[ModelRegistry] Loading all models...")
        self._load_breed_model()
        self._load_face_detector()
        self._load_encoder()
        self._load_gallery()
        print("[ModelRegistry] All models loaded successfully.")

    # ------------------------------------------------------------------
    # Branch 1 : Breed classifier
    # ------------------------------------------------------------------
    def _load_breed_model(self):
        # Load breed label list
        # FILE: models/breed_labels.txt
        with open(settings.BREED_LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        self.idx_to_breed = {i: lbl for i, lbl in enumerate(labels)}

        num_classes = len(labels)
        model = _build_breed_model(num_classes)

        # FILE: models/best_dog_breed_model_convnext.pth
        ckpt = torch.load(
            settings.BREED_MODEL_PATH,
            map_location=settings.DEVICE,
            weights_only=True,
        )
        # Support both raw state-dict and wrapped checkpoints
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.to(settings.DEVICE).eval()
        self.breed_model = model
        print(f"  [breed]    loaded {num_classes} classes from {settings.BREED_MODEL_PATH}")

    # ------------------------------------------------------------------
    # Branch 2a : YOLO dog-face detector
    # ------------------------------------------------------------------
    def _load_face_detector(self):
        # FILE: models/best.pt
        self.face_detector = YOLO(settings.FACE_DETECTOR_PATH)
        print(f"  [detector] loaded {settings.FACE_DETECTOR_PATH}")

    # ------------------------------------------------------------------
    # Branch 2b : Identity encoder
    # ------------------------------------------------------------------
    def _load_encoder(self):
        encoder = _build_encoder(settings.EMBEDDING_DIM)

        # FILE: models/Dog_encoder_best.pth
        ckpt = torch.load(
            settings.ENCODER_MODEL_PATH,
            map_location=settings.DEVICE,
            weights_only=True,
        )
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        encoder.load_state_dict(state, strict=False)
        encoder.to(settings.DEVICE).eval()
        self.encoder = encoder
        print(f"  [encoder]  loaded {settings.ENCODER_MODEL_PATH}")

    # ------------------------------------------------------------------
    # Gallery (pre-computed embeddings + id mapping)
    # ------------------------------------------------------------------
    def _load_gallery(self):
        # FILE: models/embeddings.npy  -- shape (N, D)
        embs = np.load(settings.GALLERY_EMBEDDINGS_PATH).astype(np.float32)
        # L2-normalise once at load time for fast cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        self.gallery_embs = embs / norms

        # FILE: models/labels.npy  -- shape (N,) integer indices
        label_indices = np.load(settings.GALLERY_LABELS_PATH)

        # FILE: models/idx2class.json  -- {"0": "Bruno", "1": "Lola", ...}
        with open(settings.IDX2CLASS_PATH, "r", encoding="utf-8") as f:
            idx2class = json.load(f)

        # Convert integer label indices -> identity name strings
        self.gallery_ids = [
            idx2class[str(int(idx))] for idx in label_indices
        ]
        print(
            f"  [gallery]  {len(self.gallery_ids)} enrolled embeddings "
            f"from {settings.GALLERY_EMBEDDINGS_PATH}"
        )
