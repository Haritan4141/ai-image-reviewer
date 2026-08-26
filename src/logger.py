"""Application logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    """Configure console and UTF-8 file logging once, then return the app logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ai_image_reviewer")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
