#!/usr/bin/env python3
"""Entrypoint script to run flows and anything easily through commands.

Usage example:

```bash
# Prepare training data
python main.py prepare-train-data

# Use --help to see all options
python main.py --help
```
"""

import typer
import hydra
import pickle
from omegaconf import OmegaConf

from housing_prediction.dataset import load_data
from housing_prediction.features import (
    feature_engineering_train,
    feature_engineering_test,
)
from housing_prediction.train import train
from housing_prediction.predict import predict
from housing_prediction.models import load_model
from housing_prediction.utils import read_tabular_data
from housing_prediction.config import config_logger, DATA_DIR, ARTIFACTS_DIR

app = typer.Typer()


def get_cfg():
    hydra.initialize("config", version_base=None)
    cfg = OmegaConf.to_container(hydra.compose(config_name="train"))
    return cfg


@app.command()
def prepare_train_data():
    """Prepare training data and save data artifacts"""
    # Setup configuration
    config_logger()
    # cfg = get_cfg()

    # Load data
    input_data = DATA_DIR / "raw/train.csv"
    df = load_data(input_data)
    df, feature_cols, artifacts = feature_engineering_train(
        df, output_path=DATA_DIR / "processed/train_features.csv"
    )

    # Save feature cols and artifacts for inference later
    with open(DATA_DIR / "processed/feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    with open(ARTIFACTS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)


@app.command()
def prepare_test_data():
    """Prepare test data and use data artifacts"""
    # Setup configuration
    config_logger()
    # cfg = get_cfg()

    # Load data
    input_data = DATA_DIR / "raw/test.csv"
    df = load_data(input_data)

    # Load artifacts
    with open(ARTIFACTS_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    df = feature_engineering_test(
        df, artifacts, output_path=DATA_DIR / "processed/test_features.csv"
    )


@app.command()
def training():
    """Train model"""
    # Setup configuration
    config_logger()
    cfg = get_cfg()

    # Load preprocessed data
    df = read_tabular_data(DATA_DIR / "processed/train_features.csv")
    with open(DATA_DIR / "processed/feature_cols.txt", "r") as f:
        feature_cols = f.read().split("\n")

    model, oof_preds, evaluation_results = train(
        X=df[feature_cols],
        y=df[cfg["target_col"]],
        model_params=cfg["model_params"],
        cv=cfg["cv"],
        metrics=cfg["metrics"],
        model_path=ARTIFACTS_DIR / "model.pkl",
    )


@app.command()
def inference():
    """Make predictions"""
    # Setup configuration
    config_logger()
    # cfg = get_cfg()

    # Load artifacts
    with open(DATA_DIR / "processed/feature_cols.txt", "r") as f:
        feature_cols = f.read().split("\n")

    # Load features
    df = read_tabular_data(DATA_DIR / "processed/test_features.csv")

    # Load model
    model = load_model(ARTIFACTS_DIR / "model.pkl")

    # Inference
    _ = predict(
        df[feature_cols],
        model=model,
        output_path=DATA_DIR / "processed/predictions.csv",
    )


if __name__ == "__main__":
    app()
