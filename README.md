# Dog Face Recognition

Breed classification + identity matching pipeline built with FastAPI, PyTorch, and Ultralytics YOLO.

## Project Structure

```
Dog-Face-Recognition/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── core/
│   │   └── config.py              # All settings and model paths
│   ├── services/
│   │   ├── model_registry.py      # Loads all models at startup
│   │   ├── breed_service.py       # Breed classification inference
│   │   ├── face_service.py        # YOLO face detection and crop
│   │   └── identity_service.py   # Embedding extraction and gallery match
│   └── schemas/
│       └── schemas.py             # Pydantic response model
├── frontend/
│   └── index.html                 # Upload UI
├── models/                        # <-- Place your model files here
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Required Model Files

Place these files inside the `models/` folder before running:

| Filename | Description |
|---|---|
| `best_dog_breed_model_convnext.pth` | ConvNeXt breed classifier weights |
| `breed_labels.txt` | One breed label per line, sorted order matching training |
| `best.pt` | YOLO dog-face detector weights |
| `Dog_encoder_best.pth` | Identity feature encoder weights |
| `embeddings.npy` | Gallery face embeddings (from stage3/) |
| `labels.npy` | Gallery label indices (from stage3/) |
| `idx2class.json` | Index to dog identity name map (from stage3/) |

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint.

Open `frontend/index.html` in your browser, set the API URL to `http://127.0.0.1:8000`, and upload a dog image.

## API Response

```json
{
  "breed": "Golden Retriever",
  "breed_confidence": 0.9821,
  "face_detected": true,
  "detector_confidence": 0.8934,
  "dog_id": "Bruno",
  "match_score": 0.9134,
  "bbox": [72, 54, 218, 209],
  "crop_image_base64": "...",
  "message": "Matched dog: Bruno"
}
```

## Deploy on Render

- Root Directory: `.` (repo root)
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add env var `FRONTEND_ORIGIN` = your GitHub Pages URL

## Tune Identity Threshold

The `MATCH_THRESHOLD` in `app/core/config.py` (default `0.60`) controls when a dog is
marked `unknown` versus matched. Raise it to be stricter, lower it to be more permissive.
