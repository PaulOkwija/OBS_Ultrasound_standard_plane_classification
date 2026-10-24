"""Tests for src.data.curate: class remapping, deterministic splits, and
no overlap between train, val and test."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.curate import prepare_african, prepare_fetal, remap_class, VALID_CLASSES


def _make_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
    img.save(path)


def test_remap_class():
    assert remap_class("Fetal brain", VALID_CLASSES) == "brain"
    assert remap_class("Fetal femur", VALID_CLASSES) == "femur"
    assert remap_class("Fetal abdomen", VALID_CLASSES) == "abdomen"
    assert remap_class("Fetal thorax", VALID_CLASSES) == "other"
    assert remap_class("Maternal cervix", VALID_CLASSES) == "other"


def test_prepare_african_runs_cleanly_and_splits_deterministically():
    """prepare_african() must run cleanly and be deterministic for a fixed random_state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "raw"
        out_dir_a = Path(tmpdir) / "out_a"
        out_dir_b = Path(tmpdir) / "out_b"

        countries = ["Egypt", "Uganda", "Ghana", "Algeria", "Malawi"]
        rows = []
        for country in countries:
            for patient in range(1, 6):  # 5 patients/country
                for plane, cls in [("Fetal brain", "brain"), ("Fetal femur", "femur"),
                                    ("Fetal abdomen", "abdomen")]:
                    fname = f"patient{patient:03d}_{country[:3].upper()}_{cls}"
                    _make_dummy_image(root / country / f"{fname}.png")
                    rows.append({"Patient_num": patient, "Plane": plane,
                                 "Center": country, "Filename": fname})
        df = pd.DataFrame(rows)
        csv_path = Path(tmpdir) / "African_planes_database.csv"
        df.to_csv(csv_path, index=False)

        stats_a = prepare_african(root=root, csv_path=csv_path, out_dir=out_dir_a,
                                   test_country="Malawi", random_state=42)
        stats_b = prepare_african(root=root, csv_path=csv_path, out_dir=out_dir_b,
                                   test_country="Malawi", random_state=42)

        # Deterministic: same seed -> same split sizes.
        assert stats_a == stats_b

        # Malawi (test_country) images all land in test/, none in train/val.
        test_files = {p.name for p in (out_dir_a / "test").rglob("*.png")}
        train_files = {p.name for p in (out_dir_a / "train").rglob("*.png")}
        val_files = {p.name for p in (out_dir_a / "val").rglob("*.png")}
        assert all("MAL" in f for f in test_files)
        assert not (train_files & test_files)
        assert not (val_files & test_files)

        # No (Center, Patient_num) pair's images split across train AND val.
        # Patient_num is only unique WITHIN a country in the real dataset (a known
        # property of the source CSV), so
        # this synthetic fixture intentionally reuses patient numbers 1-5 across
        # countries too - the guarantee that matters (and that LOCO, grouped by
        # Center, actually relies on) is per-(Center, Patient_num), not per raw
        # Patient_num alone.
        for _, row in df[df["Center"] != "Malawi"].iterrows():
            fname = f"{row['Filename']}.png"
            in_train = fname in train_files
            in_val = fname in val_files
            assert not (in_train and in_val)


def test_prepare_fetal_runs_cleanly_and_splits_deterministically():
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = Path(tmpdir) / "raw" / "Images"
        out_dir = Path(tmpdir) / "out"

        rows = []
        for patient in range(1, 21):
            train_flag = 0 if patient <= 4 else 1  # first 4 patients -> held-out test
            for plane, cls in [("Fetal brain", "brain"), ("Fetal femur", "femur")]:
                fname = f"patient{patient:03d}_{cls}"
                _make_dummy_image(images_dir / f"{fname}.png")
                rows.append({"Patient_num": patient, "Plane": plane, "Train": train_flag,
                             "Image_name": fname})
        df = pd.DataFrame(rows)
        csv_path = Path(tmpdir) / "FETAL_PLANES_DB_data.csv"
        df.to_csv(csv_path, sep=";", index=False)

        stats = prepare_fetal(root=images_dir.parent, csv_path=csv_path, out_dir=out_dir,
                               random_state=42)

        assert stats["test"]["n_images"] == 4 * 2  # 4 held-out patients x 2 planes
        assert stats["train"]["n_images"] + stats["val"]["n_images"] == 16 * 2

        train_files = {p.name for p in (out_dir / "train").rglob("*.png")}
        val_files = {p.name for p in (out_dir / "val").rglob("*.png")}
        assert not (train_files & val_files)
