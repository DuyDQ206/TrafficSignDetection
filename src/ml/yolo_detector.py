from pathlib import Path
from ultralytics import YOLO
import numpy as np


class YOLODetector:
    def __init__(self, model_path: str):
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")

        # Load model once when the detector is initialized
        self.model = YOLO(str(path))

    def predict(self, image: np.ndarray, conf: float = 0.25):
        results = self.model.predict(
            source=image,
            conf=conf,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })

        return detections