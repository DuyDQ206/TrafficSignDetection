from fastapi import APIRouter
from src.backend.api.v1 import detection, health

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(detection.router)