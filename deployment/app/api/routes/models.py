"""
Model Management Routes
-----------------------
GET  /api/v1/models/available  - list all models in registry
GET  /api/v1/models/loaded     - list currently loaded models
POST /api/v1/models/load       - load a model into memory
POST /api/v1/models/unload     - unload a model from memory
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.model_manager import model_manager

router = APIRouter()


class LoadRequest(BaseModel):
    name: str
    version: str = "latest"


class UnloadRequest(BaseModel):
    name: str
    version: str = "latest"


@router.get("/available")
def list_available_models():
    """List all model versions present in the model registry directory."""
    return {"models": model_manager.list_available()}


@router.get("/loaded")
def list_loaded_models():
    """List all models currently loaded in memory."""
    return {"models": model_manager.list_loaded()}


@router.post("/load")
def load_model(req: LoadRequest):
    """
    Load a model into memory by name and version.
    If the model is already loaded, this is a no-op.
    """
    try:
        m = model_manager.load_model(req.name, req.version)
        return {
            "status": "loaded",
            "name": m.name,
            "version": m.version,
            "device": str(m.device),
            "num_classes": m.num_classes,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/unload")
def unload_model(req: UnloadRequest):
    """Unload a model from memory to free resources."""
    try:
        model_manager.unload_model(req.name, req.version)
        return {"status": "unloaded", "name": req.name, "version": req.version}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
