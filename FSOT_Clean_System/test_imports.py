#!/usr/bin/env python3
"""
Import Test File - Auto-generated for Pylance debugging
This file should import successfully if all dependencies are working
"""

# Test basic imports
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Test optional web dependencies
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    print("SUCCESS: FastAPI imports successful")
except ImportError as e:
    print(f"FAILED: FastAPI import failed: {e}")

# Test testing framework
try:
    import pytest
    print("SUCCESS: pytest import successful")
except ImportError as e:
    print(f"FAILED: pytest import failed: {e}")

# Test project modules
try:
    from core.fsot_engine import FSOTEngine
    from brain.brain_orchestrator import BrainOrchestrator
    print("SUCCESS: Project modules import successful")
except ImportError as e:
    print(f"FAILED: Project modules import failed: {e}")

if __name__ == "__main__":
    print("Import test completed")
