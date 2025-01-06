"""Setup configuration for pathing and logging."""

from pathlib import Path

from loguru import logger

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

LOG_PATH = PROJ_ROOT / "logs/file.log"
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)


def config_logger():
    """Configure the logger to log to console and file."""
    # If tqdm is installed, configure loguru with tqdm.write
    # https://github.com/Delgan/loguru/issues/135
    try:
        from tqdm import tqdm

        logger.remove()  # Remove the default handler.
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass

    logger.add(LOG_PATH, level="INFO", rotation="10 MB")  # Also log to a file
