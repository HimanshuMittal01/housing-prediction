"""Predict module."""

from loguru import logger

from housing_prediction.utils import read_tabular_data, write_tabular_data


def predict(X, model, output_path=None):
    """Make predictions using a model.

    Args:
        X (DataFrame): Features.
        model (object): Model object.
        output_path (Path, optional): Path to save the predictions. Defaults to None.
    """
    # Make predictions
    predictions = model.predict(X)

    # Save the predictions
    if output_path is not None:
        write_tabular_data(predictions, output_path)
        logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    from housing_prediction.config import config_logger, DATA_DIR

    config_logger()

    # Load model
    from housing_prediction.models import load_model

    model = load_model(DATA_DIR / "models/model.pkl")

    # Load feature cols
    with open(DATA_DIR / "processed/feature_cols.txt", "r") as f:
        feature_cols = f.read().splitlines()

    # Load dataset
    test_data = read_tabular_data(DATA_DIR / "processed/test_features.csv")

    # Make predictions
    predict(
        test_data[feature_cols], model, DATA_DIR / "processed/predictions.csv"
    )
