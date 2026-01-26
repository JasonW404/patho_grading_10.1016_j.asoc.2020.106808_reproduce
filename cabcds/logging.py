"""Logging configuration for the project."""
import logging
import sys

from cabcds.config import config

def setup_logger(name: str = "cabcds", level: str = "INFO" if not config.debug else "DEBUG") -> logging.Logger:
    """Configure and return a logger with a standard format."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"cabcds.{name}")
