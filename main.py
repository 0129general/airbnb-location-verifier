"""
Main entry point for Airbnb Location Verifier
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.core.app import app, initialize_app
from src.config import config

# Initialize app when module is loaded (for gunicorn)
initialize_app()

if __name__ == "__main__":
    # Only run in debug mode when explicitly executed
    # For production deployment, gunicorn will import this module
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║     Airbnb Location Verifier v1.0.0         ║
    ║     Starting Flask application...            ║
    ╚══════════════════════════════════════════════════╝
    
    Configuration:
    - Debug Mode: {config.DEBUG}
    - AI Features: {config.ENABLE_AI_FEATURES}
    - Google Vision: {config.ENABLE_GOOGLE_VISION}
    - Real Estate Search: {config.ENABLE_REAL_ESTATE_SEARCH}
    - Environment: {os.environ.get('FLASK_ENV', 'development')}
    """)
    
    # Only use debug mode when running directly (development)
    # Production will use gunicorn which doesn't need this
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG)