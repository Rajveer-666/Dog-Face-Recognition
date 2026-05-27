# =============================================================
# app/main.py  --  FastAPI entry point
# =============================================================

import io
import base64
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.core.config import settings
from app.services.model_registry import ModelRegistry
from app.services.breed_service import predict_breed
from app.services.face_service import detect_and_crop_face
from app.services.identity_service import extract_embedding, match_identity
from app.schemas.schemas import PredictResponse

# ---------- global model registry ----------
registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup."""
    registry.load_all()
    yield
    # cleanup if needed


app = FastAPI(
    title="Dog Face Recognition API",
    description="Breed classification + dog identity matching pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------- CORS ----------
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------
def read_upload_as_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def pil_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def encode_crop_base64(crop_bgr: np.ndarray) -> str:
    """Encode a BGR crop to base64 JPEG string for frontend display."""
    _, buf = cv2.imencode(".jpg", crop_bgr)
    return base64.b64encode(buf).decode("utf-8")


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": settings.DEVICE}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # ---- validate upload ----
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_bytes = await file.read()
    pil_img = read_upload_as_pil(file_bytes)
    cv_img  = pil_to_cv2_bgr(pil_img)

    # ---- BRANCH 1 : Breed classification (full image) ----
    breed_name, breed_conf = predict_breed(pil_img, registry)

    # ---- BRANCH 2a : Dog face detection + crop ----
    face_crop, bbox, det_conf = detect_and_crop_face(cv_img, registry)

    if face_crop is None:
        return PredictResponse(
            breed=breed_name,
            breed_confidence=round(breed_conf, 4),
            face_detected=False,
            detector_confidence=None,
            dog_id=None,
            match_score=None,
            bbox=None,
            crop_image_base64=None,
            message="Breed predicted, but no dog face detected — identity matching skipped.",
        )

    # ---- BRANCH 2b : Feature extraction + identity matching ----
    embedding          = extract_embedding(face_crop, registry)
    dog_id, match_score = match_identity(embedding, registry)

    crop_b64 = encode_crop_base64(face_crop)

    message = (
        f"Matched dog: {dog_id}"
        if dog_id != "unknown"
        else "No matching dog found in gallery (score below threshold)."
    )

    return PredictResponse(
        breed=breed_name,
        breed_confidence=round(breed_conf, 4),
        face_detected=True,
        detector_confidence=round(det_conf, 4) if det_conf else None,
        dog_id=dog_id,
        match_score=round(match_score, 4),
        bbox=bbox,
        crop_image_base64=crop_b64,
        message=message,
    )
