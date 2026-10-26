import time
import cv2
import tempfile
import shutil
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from src.backend.config import OPENVINO_MODEL_PATH, YOLO_MODEL_PATH
from src.backend.schemas.detection_schema import PredictImageResponse, PredictVideoResponse
from src.ml.yolo_detector import YOLODetector

router = APIRouter(tags=["Detection"])

# ── Khởi tạo detector một lần duy nhất khi server start ──
detector = YOLODetector(
    openvino_path=str(OPENVINO_MODEL_PATH),
    pt_path=str(YOLO_MODEL_PATH),
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime"}
SKIP_FRAMES = 3  # Nhận diện mỗi 3 frame (tối ưu CPU)


# ── Endpoint 1: Nhận diện ảnh ──────────────────────────────
@router.post("/predict-image", response_model=PredictImageResponse)
async def predict_image(
    file: UploadFile = File(..., description="Ảnh JPEG / PNG / WebP"),
    conf: float = 0.25,
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Chỉ chấp nhận JPEG, PNG, WebP.")

    if not 0 <= conf <= 1:
        raise HTTPException(status_code=400, detail="conf phải trong khoảng [0, 1].")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Ảnh không được vượt quá 10 MB.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ hoặc bị hỏng.")

    height, width = image.shape[:2]

    try:
        start = time.perf_counter()
        detections = detector.predict(image, conf=conf)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference thất bại.") from exc

    return PredictImageResponse(
        success=True,
        model_backend=detector.backend,
        image={"filename": file.filename, "width": width, "height": height},
        inference_time_ms=round(elapsed_ms, 2),
        detection_count=len(detections),
        detections=detections,
    )


# ── Endpoint 2: Nhận diện video ────────────────────────────
@router.post("/predict-video", response_model=PredictVideoResponse)
async def predict_video(
    file: UploadFile = File(..., description="Video MP4 / AVI / MOV"),
    conf: float = 0.25,
):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=415, detail="Chỉ chấp nhận MP4, AVI, MOV.")

    video_bytes = await file.read()

    if not video_bytes:
        raise HTTPException(status_code=400, detail="File video rỗng.")

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="Video không được vượt quá 100 MB.")

    # Lưu tạm ra đĩa để OpenCV đọc
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detections: list[dict] = []
        frame_index = 0
        last_boxes: list[dict] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            timestamp_ms = int((frame_index / fps) * 1000)

            # Inference mỗi SKIP_FRAMES frame
            if frame_index % SKIP_FRAMES == 1:
                resized = cv2.resize(frame, (640, 640))
                results = detector.predict(resized, conf=conf)
                last_boxes = []
                for det in results:
                    # Chuyển bbox tuyệt đối → normalized xywh
                    x1, y1, x2, y2 = det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]
                    bx = (x1 + x2) / 2 / 640
                    by = (y1 + y2) / 2 / 640
                    bw = (x2 - x1) / 640
                    bh = (y2 - y1) / 640
                    last_boxes.append({
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox_normalized": {
                            "x_center": round(bx, 4),
                            "y_center": round(by, 4),
                            "width": round(bw, 4),
                            "height": round(bh, 4),
                        },
                    })

            if last_boxes:
                detections.append({
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "objects": last_boxes,
                })

        cap.release()
    finally:
        import os
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return PredictVideoResponse(
        success=True,
        model_backend=detector.backend,
        video_info={"fps": fps, "total_frames": total_frames, "duration_seconds": round(total_frames / fps, 2)},
        detection_count=sum(len(f["objects"]) for f in detections),
        detections=detections,
    )