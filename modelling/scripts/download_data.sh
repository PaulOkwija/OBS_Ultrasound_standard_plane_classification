#!/bin/bash
# Prepare the African and FetalPlanes DB datasets from Hugging Face Hub.
# Run from modelling/:  bash scripts/download_data.sh
#
# Downloads the raw source archives (mirrored from their original Zenodo
# records, both CC BY 4.0) and re-derives the train/val/test splits locally -
# see scripts/prepare_datasets.py and src/data/curate.py.
#
# Output structure:
#   data/african/train|val|test/brain|femur|abdomen|other/
#   data/fetal_db/train|val|test/brain|femur|abdomen|other/

set -e

echo "=== Installing huggingface_hub ==="
pip install huggingface_hub -q

echo ""
echo "=== Preparing datasets (download + curate) ==="
python scripts/prepare_datasets.py "$@"
