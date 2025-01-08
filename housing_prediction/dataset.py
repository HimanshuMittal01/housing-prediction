"""Load and clean data, manage data streams and handle holdout splits.

Apache Software License 2.0
Copyright (c) 2025, Himanshu Mittal"""

import polars as pl
from loguru import logger

from housing_prediction.utils import write_tabular_data


def load_data(
    input_path,
    output_path=None,
):
    """Load and clean the raw dataset.

    Args:
        input_path (Path): Path to the raw dataset.
        output_path (Path, optional): Path to save the processed dataset. Defaults to None.

    Returns:
        DataFrame: Cleaned training dataset.
    """
    logger.info("Processing dataset...")

    ### Preprocessing START ###

    # Read the dataset
    df = pl.read_csv(input_path, null_values=["N/A", "null"])

    # Clean the price column
    df = df.with_columns(
        pl.col("price")
        .replace(
            {
                "$279,000+": "279000",
                "Est. $138.8K": "138800",
                "Est. $290K": "290000",
            }
        )
        .str.replace("$", "", literal=True)
        .str.replace_all(",", "")
        .cast(pl.Float32)
    )

    # Location of the property also matters, let's extract the zip code from the address
    df = df.with_columns(
        pl.col("address").str.extract(r"IL (\d{5})$").alias("zipcode")
    )

    # Fill in the missing zip code (found using Google Search)
    df = df.with_columns(
        zipcode=pl.when(pl.col("address") == "Madison FP Plan, Madison")
        .then(pl.lit("60601"))
        .otherwise(pl.col("zipcode"))
    )

    # Clean the square footage column, convert unknown values to null
    df = df.with_columns(
        pl.col("square_footage").cast(pl.Float32, strict=False)
    )

    # Remove outliers
    df = df.filter(
        ~pl.col("address").is_in(
            [
                "1355 N Astor St, Chicago, IL 60610",
                "415 E North Water St #3205, Chicago, IL 60611",
            ]
        )  # extremely high sq ft given less number of bedrooms
    ).filter(
        pl.col("zipcode")
        != "60602"  # missing bedrooms for all properties in this zip code (2 rows)
    )

    ### Preprocessing END ###

    logger.success("Processing dataset complete.")

    if output_path is not None:
        write_tabular_data(df, output_path)

    return df


if __name__ == "__main__":
    from housing_prediction.config import config_logger, DATA_DIR

    config_logger()

    _ = load_data(
        DATA_DIR / "raw/dataset.csv", DATA_DIR / "processed/cleaned_train.csv"
    )
