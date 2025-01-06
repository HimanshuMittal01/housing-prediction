"""Utility functions"""

from pathlib import Path

import polars as pl
from loguru import logger


def read_tabular_data(input_path):
    """Read a file into a DataFrame.

    Args:
        input_path (Path): Path to the file to read.

    Returns:
        DataFrame: DataFrame from the file.
    """
    input_path = Path(input_path)
    if input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    elif input_path.suffix == ".xlsx":
        df = pl.read_excel(input_path)
    else:
        raise ValueError("File format not supported.")
    logger.info(f"DataFrame loaded from {input_path}")
    return df


def write_tabular_data(df, output_path, overwrite=True):
    """Write a DataFrame to a file.

    Args:
        df (DataFrame): DataFrame to write.
        output_path (Path): Path to save the DataFrame.
        overwrite (bool, optional): Overwrite the file if it exists. Defaults to True.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set `overwrite=True` to overwrite the file."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    logger.info(f"DataFrame saved to {output_path}")
