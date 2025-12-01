"""
WSGI entry point for production deployment with gunicorn
"""
import os
import sys
from pathlib import Path

# Set production environment
os.environ['FLASK_ENV'] = 'production'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.core.app import app, initialize_app

# Initialize the application for production
initialize_app()

# Export app for gunicorn
application = app

if __name__ == "__main__":
    # This should not be run directly in production
    print("Warning: This file is meant to be run with gunicorn, not directly")
    print("Use: gunicorn wsgi:application")