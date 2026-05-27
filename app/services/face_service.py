# =============================================================
# app/services/face_service.py  --  Dog face detection & crop
# =============================================================
# Input  : OpenCV BGR image (full upload)
# Output : (face_crop_bgr, bbox, detector_confidence)
#          Returns (None, None, None) if no face detected.
# =============================================================

from typing import Optional, Tuple, List

import cv2
import numpy as np

from app.core.config import settings


def detect_and_crop_face(
    cv_img: np.ndarray,
    registry,
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[float]]:
    """
    Run YOLO dog-face detector on the full image, pick the highest-
    confidence detection, expand the bounding box with notebook padding,
    and return the cropped face region.

    Padding values match the face-detection notebook exactly:
        PAD_TOP = 0.18
        PAD_X   = 0.13
        PAD_BOT = 0.10

    Args:
        cv_img   : OpenCV BGR numpy array (H, W, 3)
        registry : ModelRegistry holding face_detector (Ultralytics YOLO)

    Returns:
        face_crop : BGR crop of the detected dog face  (or None)
        bbox      : [x1, y1, x2, y2] padded box in original image coords
        det_conf  : detector confidence score (0-1)
    """
    h, w = cv_img.shape[:2]

    results = registry.face_detector(
        cv_img,
        conf=settings.DET_CONF,
        verbose=False,
    )

    best_box  = None
    best_conf = -1.0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()   # (M, 4)  x1 y1 x2 y2
        confs = r.boxes.conf.cpu().numpy()   # (M,)
        for box, conf in zip(boxes, confs):
            if float(conf) > best_conf:
                best_conf = float(conf)
                best_box  = box[:4]

    if best_box is None:
        return None, None, None

    x1, y1, x2, y2 = best_box
    bw = x2 - x1
    bh = y2 - y1

    # Expand box with notebook padding values
    nx1 = max(0, int(x1 - settings.PAD_X   * bw))
    nx2 = min(w, int(x2 + settings.PAD_X   * bw))
    ny1 = max(0, int(y1 - settings.PAD_TOP  * bh))
    ny2 = min(h, int(y2 + settings.PAD_BOT  * bh))

    face_crop = cv_img[ny1:ny2, nx1:nx2]
    bbox      = [nx1, ny1, nx2, ny2]

    return face_crop, bbox, best_conf
