"""Data preprocessing, transformation, imputation, and feature engineering."""

import polars as pl
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

    df = df.with_columns(
        (pl.col("bedrooms") + pl.col("bathrooms")).alias("B_plus_B"),
        (pl.col("bedrooms") * pl.col("bathrooms")).alias("B_prod_B"),
        (pl.col("bedrooms") / pl.col("bathrooms")).alias("B_div_B"),
        (pl.col("square_footage") / pl.col("bedrooms")).alias("sq_div_bed"),
        (pl.col("square_footage") / pl.col("bathrooms")).alias("sq_div_bath"),
        (pl.col("square_footage").median().over("zipcode")).alias(
            "median_sq_ft_zipcode"
        ),
        (pl.col("square_footage").mean().over("zipcode")).alias(
            "mean_sq_ft_zipcode"
        ),
        (pl.col("square_footage").std().over("zipcode")).alias(
            "std_sq_ft_zipcode"
        ),
        (pl.col("square_footage").min().over("zipcode")).alias(
            "min_sq_ft_zipcode"
        ),
        (pl.col("square_footage").max().over("zipcode")).alias(
            "max_sq_ft_zipcode"
        ),
        (pl.col("price").median().over("zipcode")).alias(
            "median_price_zipcode"
        ),
        (pl.col("price").mean().over("zipcode")).alias("mean_price_zipcode"),
        (pl.col("price").std().over("zipcode")).alias("std_price_zipcode"),
        (pl.col("price").min().over("zipcode")).alias("min_price_zipcode"),
        (pl.col("price").max().over("zipcode")).alias("max_price_zipcode"),
        (
            (pl.col("price") / pl.col("square_footage"))
            .median()
            .over("zipcode")
        ).alias("median_price_per_sq_ft_zipcode"),
        (
            (pl.col("price") / pl.col("square_footage")).mean().over("zipcode")
        ).alias("mean_price_per_sq_ft_zipcode"),
        (
            (pl.col("price") / pl.col("square_footage")).std().over("zipcode")
        ).alias("std_price_per_sq_ft_zipcode"),
        (
            (pl.col("price") / pl.col("square_footage")).min().over("zipcode")
        ).alias("min_price_per_sq_ft_zipcode"),
        (
            (pl.col("price") / pl.col("square_footage")).max().over("zipcode")
        ).alias("max_price_per_sq_ft_zipcode"),
    )

    # One hot encode zip code
    # X.drop_in_place('zipcode')
    df = df.to_dummies("zipcode")

    # Define feature columns
    feature_cols = (
        ["bedrooms", "bathrooms", "square_footage"]
        + [col for col in df.columns if col.startswith("zipcode")]
        + [
            "B_plus_B",
            "B_prod_B",
            "B_div_B",
            "sq_div_bed",
            "sq_div_bath",
            "median_sq_ft_zipcode",
            "mean_sq_ft_zipcode",
            "std_sq_ft_zipcode",
            "min_sq_ft_zipcode",
            "max_sq_ft_zipcode",
            "median_price_zipcode",
            "mean_price_zipcode",
            "std_price_zipcode",
            "min_price_zipcode",
            "max_price_zipcode",
            "median_price_per_sq_ft_zipcode",
            "mean_price_per_sq_ft_zipcode",
            "std_price_per_sq_ft_zipcode",
            "min_price_per_sq_ft_zipcode",
            "max_price_per_sq_ft_zipcode",
        ]
    )

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
