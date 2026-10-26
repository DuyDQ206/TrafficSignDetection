from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int


class PredictImageResponse(BaseModel):
    success: bool
    model_backend: str          # "openvino" hoặc "pytorch"
    image: ImageInfo
    inference_time_ms: float
    detection_count: int
    detections: list[Detection]


# --- Video schemas ---

class BBoxNormalized(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class VideoDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_normalized: BBoxNormalized


class FrameResult(BaseModel):
    frame_index: int
    timestamp_ms: int
    objects: list[VideoDetection]


class VideoInfo(BaseModel):
    fps: float
    total_frames: int
    duration_seconds: float


class PredictVideoResponse(BaseModel):
    success: bool
    model_backend: str
    video_info: VideoInfo
    detection_count: int
    detections: list[FrameResult]