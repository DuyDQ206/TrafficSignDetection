import time
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from src.backend.config import YOLO_MODEL_PATH
from src.ml.yolo_detector import YOLODetector

router = APIRouter(tags=["Detection"])

# Initialize once, not once per request
detector = YOLODetector(str(YOLO_MODEL_PATH))

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = 0.25
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WebP images are supported."
        )

    if not 0 <= conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="conf must be between 0 and 1."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    height, width = image.shape[:2]

    try:
        start_time = time.perf_counter()
        detections = detector.predict(image, conf=conf)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        ) from exc

    return {
        "success": True,
        "model": "yolov11",
        "image": {
            "filename": file.filename,
            "width": width,
            "height": height
        },
        "inference_time_ms": round(inference_time_ms, 2),
        "detection_count": len(detections),
        "detections": detections
    }