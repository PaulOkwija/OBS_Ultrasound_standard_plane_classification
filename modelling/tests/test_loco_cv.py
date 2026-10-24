"""Tests for LOCO-CV transfer learning components."""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from PIL import Image

from src.data import AfricanDatasetFromCSV, CLASSES


def test_african_dataset_from_csv():
    """Verify that AfricanDatasetFromCSV correctly handles dataframes, labels, and filters out 'other'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create dummy images
        img_paths = [
            "img1_brain.png",
            "img2_femur.png",
            "img3_unknown.png",
            "img4_abdomen.png"
        ]
        
        for p in img_paths:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(root / p)
            
        # Create dummy dataframe
        df = pd.DataFrame({
            "Filename": img_paths,
            "Plane": ["brain", "femur", "other", "abdomen"],
            "Center": ["CountryA", "CountryA", "CountryB", "CountryC"]
        })
        
        # Build dataset expecting only 3 classes
        classes_3c = ["brain", "femur", "abdomen"]
        ds = AfricanDatasetFromCSV(
            df=df,
            img_dir=str(root),
            label_col="Plane",
            filename_col="Filename",
            classes=classes_3c
        )
        
        # The 'other' row should be filtered out
        assert len(ds) == 3
        
        # Verify labels match class2idx correctly
        class2idx = {c: i for i, c in enumerate(classes_3c)}
        _, label0 = ds[0]
        assert label0 == class2idx["brain"]
        
        _, label1 = ds[1]
        assert label1 == class2idx["femur"]
        
        _, label2 = ds[2]
        assert label2 == class2idx["abdomen"]


def test_loco_cv_folds():
    """Verify that LeaveOneGroupOut splits correctly by country."""
    from sklearn.model_selection import LeaveOneGroupOut
    
    # Create dummy dataframe with 4 countries (excluding test country)
    df = pd.DataFrame({
        "Filename": [f"img{i}.png" for i in range(12)],
        "Plane": ["brain"] * 12,
        "Center": ["CountryA"] * 3 + ["CountryB"] * 3 + ["CountryC"] * 3 + ["CountryD"] * 3
    })
    
    logo = LeaveOneGroupOut()
    groups = df["Center"].values
    
    folds = list(logo.split(df, groups=groups))
    
    # Needs to be exactly 4 folds
    assert len(folds) == 4
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Validation set should be exactly one country
        val_countries = df_val["Center"].unique()
        assert len(val_countries) == 1
        
        # Train set should have the other 3 countries
        train_countries = df_train["Center"].unique()
        assert len(train_countries) == 3
        
        # No overlap between train and val countries
        assert val_countries[0] not in train_countries
        
        # Total images match
        assert len(df_train) + len(df_val) == 12
