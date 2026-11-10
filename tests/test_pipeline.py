import os
import json
import pytest
import numpy as np
import pandas as pd
import yaml

def test_config_file_exists():
    assert os.path.exists("config.yaml"), "config.yaml must exist"
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "model" in cfg
    assert "quality_gate" in cfg

def test_raw_dataset_exists():
    assert os.path.exists("data/raw/churn.csv"), "churn.csv must exist in data/raw/"
    df = pd.read_csv("data/raw/churn.csv")
    assert "Churn" in df.columns
    assert len(df) > 1000

def test_processed_features_format():
    if os.path.exists("data/processed/X_train_final.npy"):
        X_train = np.load("data/processed/X_train_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        assert not np.isnan(X_train).any(), "X_train should not have NaNs"
        assert len(X_train) == len(y_train), "Features and labels count must match"
