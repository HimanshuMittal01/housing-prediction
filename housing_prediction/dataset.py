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
    df = pl.read_csv(input_path, null_values=["N/A", "null"])

    # Clean price column; remove '$' and ',' and convert to float
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
    df = df.with_columns(
        pl.col("square_footage").cast(pl.Float32, strict=False)
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
