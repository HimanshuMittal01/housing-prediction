"""Data preprocessing, transformation, imputation, and feature engineering."""

from loguru import logger

from housing_prediction.utils import read_tabular_data, write_tabular_data


def feature_engineering_train(df, output_path=None):
    """Create feature training set.

    Args:
        df: Cleaned dataframe.

    Returns:
        DataFrame: Preprocessed dataframe where `feature_cols` are ready for modeling.
        feature_cols: Feature columns to be used for modeling
        artifacts: Generated artifacts like mean/std values, one hot encoder, etc.
    """
    logger.info("Engineering features...")

    ### Feature engineering START ###
    feature_cols = []
    ### Feature engineering END ###

    logger.info("Feature engineering complete.")

    if output_path is not None:
        write_tabular_data(df, output_path)

    return df, feature_cols, {}


def feature_engineering_test(df, artifacts, output_path=None):
    """Create feature holdout set.

    Args:
        df (DataFrame): Cleaned dataframe.
        artifacts (dict[str, Any]): Artifacts if any; that were generated to process training dataset.

    Returns:
        DataFrame: Preprocessed dataframe.
    """
    ### Feature Engineering START ###

    ### Feature Engineering END ###

    if output_path is not None:
        write_tabular_data(df, output_path)

    return df


if __name__ == "__main__":
    from housing_prediction.config import config_logger, DATA_DIR

    config_logger()

    # Run for training data
    train = read_tabular_data(DATA_DIR / "processed/cleaned_train.csv")
    train, feature_cols, artifacts = feature_engineering_train(
        train, DATA_DIR / "processed/train_features.csv"
    )

    # Save feature cols
    with open(DATA_DIR / "processed/feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    # Run for test data (it can be a holdout set or a new dataset)
    # test = read_tabular_data(DATA_DIR / "processed/cleaned_test.csv")
    # test = feature_engineering_test(test, artifacts, DATA_DIR / "processed/test_features.csv")
