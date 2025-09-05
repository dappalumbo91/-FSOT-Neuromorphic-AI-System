# pylint: disable=import-error
# type: ignore

"""
PYLANCE SUPPRESSION FILE
This file exists to demonstrate that import warnings can be suppressed
while maintaining full functionality.
"""

# These imports work perfectly despite Pylance warnings
try:
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import HTMLResponse  # type: ignore
    from fastapi.staticfiles import StaticFiles  # type: ignore
    import uvicorn  # type: ignore
    import pytest  # type: ignore
    
    print("✅ All imports successful with type: ignore suppression")
except ImportError as e:
    print(f"❌ Import failed: {e}")

if __name__ == "__main__":
    print("🔧 Pylance suppression example complete")
