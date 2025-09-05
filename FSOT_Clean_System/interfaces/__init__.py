"""Interfaces package initialization"""
from .cli_interface import CLIInterface

try:
    from .web_interface import WebInterface
    WEB_AVAILABLE = True
except ImportError:
    WebInterface = None
    WEB_AVAILABLE = False

__all__ = ['CLIInterface']

if WEB_AVAILABLE:
    __all__.append('WebInterface')
