"""
Single-Image Inference Route
----------------------------
POST /api/v1/inference/predict
"""

import asyncio
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.model_manager import model_manager
from app.core.udp_bridge import udp_bridge
from app.utils.inference_engine import run_inference
from app.utils.preprocessing import load_image_from_bytes, preprocess_image

router = APIRouter()


class Prediction(BaseModel):
    label: str
    class_index: int
    confidence: float


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Prediction]


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, etc.)"),
    model_name: str = Query(..., description="Name of the model to use"),
    model_version: str = Query("latest", description="Model version (default: latest)"),
    top_k: int = Query(5, ge=1, le=100, description="Number of top predictions"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
):
    """
    Run image classification on a single uploaded image.

    - Upload an image via multipart form
    - Specify which model (and optionally version) to use
    - Returns top-K class predictions with confidence scores
    """
    # Load model (auto-loads from the registry if it is not in memory yet)
    try:
        model_obj = model_manager.get_or_load(model_name, model_version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Read and preprocess image
    try:
        raw = await file.read()
        pil_img = load_image_from_bytes(raw)
        tensor = preprocess_image(pil_img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing error: {e}")

    # Inference
    results = run_inference(model_obj, tensor, top_k=top_k, threshold=threshold)
    preds = [Prediction(**p) for p in results[0]]

    return InferenceResponse(
        model_name=model_obj.name,
        model_version=model_obj.version,
        predictions=preds,
    )


@router.websocket("/ws-stream")
async def websocket_stream(websocket: WebSocket):
    """
    Broadcast UDP-received frames to the connected WebSocket client.
    """
    await websocket.accept()
    q = await udp_bridge.subscribe()
    try:
        while True:
            try:
                data = await q.get()
                await websocket.send_bytes(data)
            except asyncio.CancelledError:
                break
    except WebSocketDisconnect:
        pass
    finally:
        udp_bridge.unsubscribe(q)
