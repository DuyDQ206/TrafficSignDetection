# 🚦 Traffic Sign Detection

Hệ thống nhận diện biển báo giao thông sử dụng **YOLOv11** kết hợp **OpenVINO** để tối ưu inference trên CPU, phục vụ qua **FastAPI REST API** với giao diện web tích hợp sẵn.

---

## ✨ Tính năng

| Tính năng | Chi tiết |
|---|---|
| 🔍 Nhận diện ảnh | Upload ảnh JPEG / PNG / WebP, trả về bounding box + confidence |
| 🎬 Nhận diện video | Upload MP4 / AVI / MOV, phân tích từng frame (skip mỗi 3 frame để tối ưu CPU) |
| ⚡ OpenVINO | Ưu tiên chạy model OpenVINO (nhanh hơn ~2–3× so với PyTorch trên CPU) |
| 🔄 Fallback | Tự động dùng `.pt` PyTorch nếu không tìm thấy OpenVINO model |
| 🌐 Frontend | Giao diện web dark-mode có sẵn tại `src/frontend/index.html` |
| 📄 API Docs | Swagger UI tích hợp tại `/docs` |

---

## 🗂️ Cấu trúc thư mục

```
TrafficSignDetection/
├── model/
│   ├── yolo_best.pt                    # Model PyTorch (fallback)
│   └── yolo_best_openvino_model/       # Model OpenVINO (ưu tiên)
│       ├── yolo_best.xml
│       └── yolo_best.bin
│
├── notebook/
│   ├── 01_eda.ipynb                    # Phân tích dữ liệu
│   ├── 02_data_preprocessing.ipynb     # Tiền xử lý dữ liệu
│   ├── 03_train_yolo.ipynb             # Huấn luyện YOLOv11
│   ├── 03_train_Faster_RCNN.ipynb      # Huấn luyện Faster R-CNN (so sánh)
│   └── 04_benchmark_comparision.ipynb  # Benchmark & so sánh model
│
├── src/
│   ├── backend/
│   │   ├── main.py                     # FastAPI app entrypoint
│   │   ├── config.py                   # Cấu hình đường dẫn model & API
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── health.py           # GET /api/v1/health
│   │   │       └── detection.py        # POST /api/v1/predict-image|video
│   │   └── schemas/
│   │       └── detection_schema.py     # Pydantic response schemas
│   ├── frontend/
│   │   └── index.html                  # Giao diện web (vanilla HTML/CSS/JS)
│   └── ml/
│       └── yolo_detector.py            # YOLODetector wrapper class
│
├── data/                               # Dataset (bị .gitignore)
├── postman/                            # Postman collections & environments
├── test.jpg                            # Ảnh mẫu để test nhanh
├── test_model.py                       # Script test model CLI
└── requirements.txt
```

---

## 🚀 Cài đặt & Chạy

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/TrafficSignDetection.git
cd TrafficSignDetection
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `torch` và `torchvision` có thể nặng (~2 GB). Nếu chỉ dùng OpenVINO, có thể bỏ hai package này và thay bằng `openvino`.

### 4. Chuẩn bị model

Đặt file model vào thư mục `model/`:

```
model/
├── yolo_best.pt                  ← PyTorch weight (tùy chọn, fallback)
└── yolo_best_openvino_model/     ← OpenVINO model (khuyến nghị)
    ├── yolo_best.xml
    └── yolo_best.bin
```

### 5. Khởi động server

```bash
uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000
```

Server chạy tại: **http://localhost:8000**

---

## 🌐 Giao diện Web

Mở trực tiếp file `src/frontend/index.html` trong trình duyệt (không cần server riêng):

```
TrafficSignDetection/src/frontend/index.html
```

Hoặc truy cập nếu bạn serve nó qua một static server.

---

## 📡 API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: **http://localhost:8000/docs** (Swagger UI)

---

### `GET /health`

Kiểm tra trạng thái server.

**Response:**
```json
{
  "status": "ok",
  "message": "Traffic Sign Detection API is running"
}
```

---

### `POST /predict-image`

Nhận diện biển báo trong một ảnh tĩnh.

| Parameter | Type | Required | Default | Mô tả |
|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Ảnh JPEG / PNG / WebP (tối đa 10 MB) |
| `conf` | `float` | ❌ | `0.25` | Ngưỡng confidence [0.0 – 1.0] |

**Response:**
```json
{
  "success": true,
  "model_backend": "openvino",
  "image": {
    "filename": "photo.jpg",
    "width": 1280,
    "height": 720
  },
  "inference_time_ms": 42.15,
  "detection_count": 2,
  "detections": [
    {
      "class_id": 3,
      "class_name": "stop",
      "confidence": 0.9124,
      "bbox": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 220.8,
        "y2": 190.4
      }
    }
  ]
}
```

---

### `POST /predict-video`

Nhận diện biển báo trong video (frame-by-frame).

| Parameter | Type | Required | Default | Mô tả |
|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Video MP4 / AVI / MOV (tối đa 100 MB) |
| `conf` | `float` | ❌ | `0.25` | Ngưỡng confidence [0.0 – 1.0] |

**Response:**
```json
{
  "success": true,
  "model_backend": "openvino",
  "video_info": {
    "fps": 30.0,
    "total_frames": 450,
    "duration_seconds": 15.0
  },
  "detection_count": 12,
  "detections": [
    {
      "frame_index": 1,
      "timestamp_ms": 33,
      "objects": [
        {
          "class_id": 1,
          "class_name": "speed_limit_50",
          "confidence": 0.8756,
          "bbox_normalized": {
            "x_center": 0.4531,
            "y_center": 0.3214,
            "width": 0.1562,
            "height": 0.2187
          }
        }
      ]
    }
  ]
}
```

> **Ghi chú:** Video được inference mỗi 3 frame (`SKIP_FRAMES = 3`) để tối ưu hiệu năng CPU, kết quả của frame trước được tái sử dụng cho các frame bị bỏ qua.

---

## 🧪 Test nhanh

### Test bằng CLI script

```bash
python test_model.py
```

Script sẽ chạy inference trên `test.jpg` và in kết quả ra console.

### Test bằng curl

```bash
# Predict ảnh
curl -X POST http://localhost:8000/api/v1/predict-image \
  -F "file=@test.jpg" \
  -F "conf=0.3"

# Health check
curl http://localhost:8000/api/v1/health
```

### Test bằng Postman

Import collection trong thư mục `postman/` vào Postman.

---

## 🧠 Kiến trúc Model

```
Input Image/Video
       │
       ▼
  YOLODetector
  ┌─────────────────────────────┐
  │  OpenVINO IR model (.xml)   │  ← Ưu tiên (nhanh hơn trên CPU)
  │  hoặc PyTorch .pt           │  ← Fallback
  └─────────────────────────────┘
       │
       ▼
  Bounding Boxes + Class + Confidence
```

- **YOLOv11** được huấn luyện trên dataset biển báo giao thông Việt Nam / quốc tế
- **OpenVINO** export giúp tăng tốc inference ~2–3× trên CPU Intel so với PyTorch
- Frame skip (mỗi 3 frame) cho video để cân bằng tốc độ và độ chính xác

---

## 📦 Dependencies

| Package | Version | Mục đích |
|---|---|---|
| `fastapi` | ≥ 0.110.0 | Web framework |
| `uvicorn` | ≥ 0.28.0 | ASGI server |
| `ultralytics` | ≥ 8.1.0 | YOLOv11 inference |
| `torch` | ≥ 2.2.0 | Deep learning backend |
| `torchvision` | ≥ 0.17.0 | Vision utilities |
| `onnxruntime` | ≥ 1.17.0 | ONNX inference |
| `opencv-python` | ≥ 4.9.0 | Đọc/xử lý ảnh & video |
| `numpy` | ≥ 1.26.0 | Xử lý mảng số |
| `Pillow` | ≥ 10.2.0 | Xử lý ảnh |
| `pydantic` | ≥ 2.6.0 | Data validation |
| `pydantic-settings` | ≥ 2.2.0 | Quản lý cấu hình |
| `python-multipart` | ≥ 0.0.9 | File upload |

---

## 📝 Notebooks

| Notebook | Mô tả |
|---|---|
| `01_eda.ipynb` | Phân tích khám phá dữ liệu (EDA) |
| `02_data_preprocessing.ipynb` | Tiền xử lý, augmentation dataset |
| `03_train_yolo.ipynb` | Huấn luyện YOLOv11 |
| `03_train_Faster_RCNN.ipynb` | Huấn luyện Faster R-CNN (baseline so sánh) |
| `04_benchmark_comparision.ipynb` | So sánh hiệu năng giữa các model |

---

## ⚙️ Cấu hình

Chỉnh sửa [`src/backend/config.py`](src/backend/config.py) để thay đổi đường dẫn model hoặc API prefix:

```python
OPENVINO_MODEL_PATH = PROJECT_ROOT / "model" / "yolo_best_openvino_model"
YOLO_MODEL_PATH     = PROJECT_ROOT / "model" / "yolo_best.pt"
API_PREFIX          = "/api/v1"
```

---

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Commit thay đổi: `git commit -m "feat: mô tả thay đổi"`
4. Push lên branch: `git push origin feature/ten-tinh-nang`
5. Tạo Pull Request

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.
