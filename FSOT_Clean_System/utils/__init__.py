"""Utility functions and helpers"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import memory manager
from .memory_manager import memory_manager

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup clean logging configuration"""
    
    # Create logger
    logger = logging.getLogger("fsot")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_query(query: str) -> bool:
    """Validate user query input"""
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    if len(query) == 0:
        return False
    
    if len(query) > 1000:  # Maximum query length
        return False
    
    return True

def format_percentage(value: float) -> str:
    """Format float as percentage string"""
    return f"{value * 100:.1f}%"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string with ellipsis if too long"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if not"""
    path.mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent
