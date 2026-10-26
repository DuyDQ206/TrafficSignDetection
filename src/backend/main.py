from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.api.router import api_router
from src.backend.config import API_PREFIX

app = FastAPI(
    title="Traffic Sign Detection API",
    description="Nhận diện biển báo giao thông bằng YOLOv11 + OpenVINO",
    version="1.0.0",
)

# CORS — cho phép frontend / Postman gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký tất cả routes
app.include_router(api_router, prefix=API_PREFIX)