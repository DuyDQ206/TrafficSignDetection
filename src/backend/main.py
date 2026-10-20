from fastapi import FastAPI
from src.backend.api.v1 import detection, health

app = FastAPI(
    title="Traffic Sign Detection API",
    description="API for traffic sign detection using deep learning models.",
    version="1.0.0"
)

app.include_router(
    health.router,
    prefix="/api/v1"
)

app.include_router(
    detection.router,
    prefix="/api/v1"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Traffic Sign Detection API",
        "docs": "/docs"
    }