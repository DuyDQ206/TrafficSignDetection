# src/backend/api/v1/health.py

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Traffic Sign Detection API is running"
    }