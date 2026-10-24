#!/bin/bash
# Download all model weights from Hugging Face Hub.
# Usage: bash scripts/download_weights.sh

set -e

DEST_DIR="checkpoints"
mkdir -p "$DEST_DIR"

echo "=== Installing huggingface_hub ==="
pip install huggingface_hub -q

echo ""
echo "=== Downloading checkpoints ==="
python scripts/fetch_checkpoints.py --out-dir "$DEST_DIR" "$@"
