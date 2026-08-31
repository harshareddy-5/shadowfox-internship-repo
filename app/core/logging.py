import logging
import sys
from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application-wide logging with consistent format."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("documind")
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


logger = setup_logging()
