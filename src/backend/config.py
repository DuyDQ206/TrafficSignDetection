from pathlib import Path

# Gốc dự án: TrafficSignDetection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model OpenVINO (ưu tiên - nhanh trên CPU)
OPENVINO_MODEL_PATH = PROJECT_ROOT / "model" / "yolo_best_openvino_model"

# Model PyTorch (fallback)
YOLO_MODEL_PATH = PROJECT_ROOT / "model" / "yolo_best.pt"

# API settings
API_PREFIX = "/api/v1"