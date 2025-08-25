"""
Centralized logging configuration for NetBot V2.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache

from ...config.settings import get_settings


@lru_cache()
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        include_timestamp: Whether to include timestamps
    """
    settings = get_settings()
    
    # Get log level from settings or parameter
    level_str = settings.logging_config.get('level', log_level).upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Create formatter
    if include_timestamp:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is setup
    setup_logging()
    
    return logging.getLogger(name)