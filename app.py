"""
Application entry point for gunicorn
This file allows gunicorn to find the Flask app in the new structure
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import the app from the new location
from src.core.app import app

# Export app for gunicorn
__all__ = ['app']