"""
Batch Inference Route
---------------------
POST /api/v1/batch/predict  - upload multiple images at once
POST /api/v1/batch/predict-dir - process a local directory (server-side path)
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.inference_engine import run_inference
from app.utils.preprocessing import (
    load_image_from_bytes,
    load_image_from_path,
    preprocess_batch,
)

router = APIRouter()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class SingleResult(BaseModel):
    filename: str
    predictions: list[dict]
    error: str | None = None


class BatchResponse(BaseModel):
    model_name: str
    model_version: str
    total: int
    results: list[SingleResult]


def _get_model(name: str, version: str):
    try:
        return model_manager.get_or_load(name, version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


@router.post("/predict", response_model=BatchResponse)
async def batch_predict_upload(
    files: list[UploadFile] = File(...),
    model_name: str = Query(...),
    model_version: str = Query("latest"),
    top_k: int = Query(5, ge=1, le=100),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
):
    """
    Classify multiple uploaded images in a single request.
    Images are processed in batches for efficiency.
    """
    model_obj = _get_model(model_name, model_version)
    results = []

    # Read all images first
    items = []
    for f in files:
        try:
            raw = await f.read()
            img = load_image_from_bytes(raw)
            items.append((f.filename, img, None))
        except Exception as e:
            items.append((f.filename, None, str(e)))

    # Process in chunks
    valid_items = [(name, img) for name, img, err in items if img is not None]
    error_items = [(name, err) for name, img, err in items if img is None]

    all_preds = {}
    for chunk in _chunked(valid_items, settings.MAX_BATCH_SIZE):
        names, imgs = zip(*chunk)
        tensor = preprocess_batch(list(imgs))
        preds = run_inference(model_obj, tensor, top_k=top_k, threshold=threshold)
        for name, pred in zip(names, preds):
            all_preds[name] = pred

    for name, img in valid_items:
        results.append(SingleResult(filename=name, predictions=all_preds.get(name, [])))
    for name, err in error_items:
        results.append(SingleResult(filename=name, predictions=[], error=err))

    return BatchResponse(
        model_name=model_obj.name,
        model_version=model_obj.version,
        total=len(results),
        results=results,
    )


@router.post("/predict-dir", response_model=BatchResponse)
def batch_predict_directory(
    directory: str = Query(..., description="Absolute path to image directory on this machine"),
    model_name: str = Query(...),
    model_version: str = Query("latest"),
    top_k: int = Query(5, ge=1, le=100),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    recursive: bool = Query(False, description="Search subdirectories"),
):
    """
    Run batch inference on all images in a local directory.
    Useful for offline bulk processing without uploading files.
    """
    model_obj = _get_model(model_name, model_version)
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    image_paths = [
        p for p in dir_path.glob(pattern)
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        return BatchResponse(
            model_name=model_obj.name,
            model_version=model_obj.version,
            total=0,
            results=[],
        )

    results = []
    for chunk_paths in _chunked(image_paths, settings.MAX_BATCH_SIZE):
        imgs, names, errors = [], [], []
        for p in chunk_paths:
            try:
                imgs.append(load_image_from_path(p))
                names.append(p.name)
                errors.append(None)
            except Exception as e:
                imgs.append(None)
                names.append(p.name)
                errors.append(str(e))

        valid = [(n, img) for n, img, e in zip(names, imgs, errors) if img is not None]
        errored = [(n, e) for n, img, e in zip(names, imgs, errors) if img is None]

        if valid:
            vnames, vimgs = zip(*valid)
            tensor = preprocess_batch(list(vimgs))
            preds = run_inference(model_obj, tensor, top_k=top_k, threshold=threshold)
            for name, pred in zip(vnames, preds):
                results.append(SingleResult(filename=name, predictions=pred))

        for name, err in errored:
            results.append(SingleResult(filename=name, predictions=[], error=err))

    return BatchResponse(
        model_name=model_obj.name,
        model_version=model_obj.version,
        total=len(results),
        results=results,
    )
