# =============================================================
# app/core/config.py  --  Central configuration
# =============================================================
# All model file paths point to the models/ directory.
# Place the following files inside models/ before running:
#
#   models/best_dog_breed_model_convnext.pth   <- breed classifier weights
#   models/breed_labels.txt                    <- one breed label per line (sorted, same order as training)
#   models/best.pt                             <- YOLO dog-face detector weights
#   models/Dog_encoder_best.pth                <- identity feature encoder weights
#   models/embeddings.npy                      <- gallery embeddings  (from stage3/)
#   models/labels.npy                          <- gallery label indices (from stage3/)
#   models/idx2class.json                      <- index -> dog identity name  (from stage3/)
# =============================================================

import os
import torch


class Settings:
    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # Model paths  (relative to repo root)
    # ------------------------------------------------------------------ #

    # -- Branch 1 : Breed classification --
    # FILE NEEDED: models/best_dog_breed_model_convnext.pth
    BREED_MODEL_PATH: str = os.getenv(
        "BREED_MODEL_PATH", "models/best_dog_breed_model_convnext.pth"
    )

    # FILE NEEDED: models/breed_labels.txt
    # One breed folder-name per line, e.g. n02085620-Chihuahua
    # Must be in the EXACT sorted order used during training.
    BREED_LABELS_PATH: str = os.getenv(
        "BREED_LABELS_PATH", "models/breed_labels.txt"
    )

    # Number of breed classes (Stanford Dogs = 120)
    NUM_BREED_CLASSES: int = int(os.getenv("NUM_BREED_CLASSES", "120"))

    # -- Branch 2a : Dog face detector --
    # FILE NEEDED: models/best.pt
    FACE_DETECTOR_PATH: str = os.getenv(
        "FACE_DETECTOR_PATH", "models/best.pt"
    )

    # Detection confidence threshold (matches notebook: CONF = 0.25)
    DET_CONF: float = float(os.getenv("DET_CONF", "0.25"))

    # Face-crop padding (matches notebook values)
    PAD_TOP: float = float(os.getenv("PAD_TOP", "0.18"))
    PAD_X:   float = float(os.getenv("PAD_X",   "0.13"))
    PAD_BOT: float = float(os.getenv("PAD_BOT", "0.10"))

    # -- Branch 2b : Identity encoder --
    # FILE NEEDED: models/Dog_encoder_best.pth
    ENCODER_MODEL_PATH: str = os.getenv(
        "ENCODER_MODEL_PATH", "models/Dog_encoder_best.pth"
    )

    # Encoder input image size
    ENCODER_IMG_SIZE: int = int(os.getenv("ENCODER_IMG_SIZE", "224"))

    # Embedding dimension (change to match your encoder output dim)
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))

    # -- Gallery files (generated from stage3/) --
    # FILE NEEDED: models/embeddings.npy
    GALLERY_EMBEDDINGS_PATH: str = os.getenv(
        "GALLERY_EMBEDDINGS_PATH", "models/embeddings.npy"
    )

    # FILE NEEDED: models/labels.npy
    GALLERY_LABELS_PATH: str = os.getenv(
        "GALLERY_LABELS_PATH", "models/labels.npy"
    )

    # FILE NEEDED: models/idx2class.json
    IDX2CLASS_PATH: str = os.getenv(
        "IDX2CLASS_PATH", "models/idx2class.json"
    )

    # ------------------------------------------------------------------ #
    # Inference thresholds
    # ------------------------------------------------------------------ #

    # Cosine-similarity threshold below which a dog is marked UNKNOWN
    # Tune this on your validation set (start around 0.55-0.65)
    MATCH_THRESHOLD: float = float(os.getenv("MATCH_THRESHOLD", "0.60"))

    # Breed classification image size (matches ConvNeXt training: 320)
    BREED_IMG_SIZE: int = int(os.getenv("BREED_IMG_SIZE", "320"))


settings = Settings()
