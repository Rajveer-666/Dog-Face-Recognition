# =============================================================
# app/schemas/schemas.py  --  Pydantic response model
# =============================================================

from typing import List, Optional
from pydantic import BaseModel


class PredictResponse(BaseModel):
    # --- Breed branch output ---
    breed: str
    breed_confidence: float

    # --- Face detection output ---
    face_detected: bool
    detector_confidence: Optional[float] = None
    bbox: Optional[List[int]] = None           # [x1, y1, x2, y2]

    # --- Identity matching output ---
    dog_id: Optional[str] = None               # from idx2class.json, or 'unknown'
    match_score: Optional[float] = None        # cosine similarity (0-1)

    # --- Face crop preview (base64 JPEG for frontend display) ---
    crop_image_base64: Optional[str] = None

    # --- Human-readable summary ---
    message: str

    class Config:
        # Allow arbitrary types so FastAPI can serialise numpy scalars cleanly
        json_encoders = {float: lambda v: round(float(v), 6)}
