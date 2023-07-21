import logging
import sys
from typing import Optional


def configure_logging(
    filename: Optional[str] = None, verbose: bool = False
) -> None:
    """
    Utility function to standardize logging format

    Args:
        filename: Name of file to which logging messages will be written
        verbose: If true, log verbosely
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()
    if filename is not None:
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(filename=filename, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
