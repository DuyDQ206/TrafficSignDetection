from pathlib import Path

# Project root: TrafficSignDetection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL_PATH = PROJECT_ROOT / "model" / "yolo_best.pt"

API_PREFIX = "/api/v1"