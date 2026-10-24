#!/usr/bin/env python3
"""
batch_infer.py
--------------
CLI tool for offline batch inference - no server required.

Usage:
    python scripts/batch_infer.py \
        --model efficientnet_b0 \
        --version v1 \
        --input /path/to/images \
        --output results.json \
        --top-k 3
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.inference_engine import run_inference
from app.utils.preprocessing import load_image_from_path, preprocess_batch

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Offline batch inference CLI")
    parser.add_argument("--model", required=True, help="Model name in registry")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--input", required=True, help="Directory or single image path")
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    # Collect image paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        pattern = "**/*" if args.recursive else "*"
        image_paths = [p for p in input_path.glob(pattern) if p.suffix.lower() in SUPPORTED]
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)

    if not image_paths:
        print("[WARN] No images found.")
        sys.exit(0)

    print(f"Found {len(image_paths)} images.")

    # Load model
    print(f"Loading model '{args.model}' version '{args.version}'...")
    model_obj = model_manager.load_model(args.model, args.version)
    print(f"Model loaded on {model_obj.device}.")

    # Run inference in batches
    all_results = []
    total = len(image_paths)

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    processed = 0
    for chunk in chunked(image_paths, args.batch_size):
        imgs, names, errors = [], [], []
        for p in chunk:
            try:
                imgs.append(load_image_from_path(p))
                names.append(str(p))
                errors.append(None)
            except Exception as e:
                imgs.append(None)
                names.append(str(p))
                errors.append(str(e))

        valid = [(n, img) for n, img, e in zip(names, imgs, errors) if img is not None]
        errored = [(n, e) for n, img, e in zip(names, imgs, errors) if img is None]

        if valid:
            vnames, vimgs = zip(*valid)
            tensor = preprocess_batch(list(vimgs))
            preds = run_inference(
                model_obj, tensor, top_k=args.top_k, threshold=args.threshold
            )
            for name, pred in zip(vnames, preds):
                all_results.append({"file": name, "predictions": pred})

        for name, err in errored:
            all_results.append({"file": name, "predictions": [], "error": err})

        processed += len(chunk)
        print(f"  {processed}/{total}", end="\r")

    print(f"\nDone. Writing results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
