"""
Configuration settings for Airbnb Location Verifier
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask configuration
class Config:
    # Basic Flask config
    SECRET_KEY = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("FLASK_ENV", "development") == "development"
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///verification_results.db")
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
    
    # Feature flags
    ENABLE_REAL_ESTATE_SEARCH = os.environ.get("ENABLE_REAL_ESTATE_SEARCH", "false").lower() in ["true", "1", "yes"]
    ENABLE_AI_FEATURES = bool(OPENAI_API_KEY)
    ENABLE_GOOGLE_VISION = bool(GOOGLE_APPLICATION_CREDENTIALS)
    ENABLE_APIFY = bool(APIFY_API_TOKEN)
    
    # Cache settings
    CACHE_DURATION_HOURS = 24
    SESSION_EXPIRY_HOURS = 24
    
    # Queue settings
    MAX_RETRY_COUNT = 3
    JOB_TIMEOUT_MINUTES = 30
    WORKER_HEARTBEAT_INTERVAL = 60  # seconds
    
    # Analysis settings
    QUICK_MODE_TIMEOUT = 10  # seconds
    QUICK_MODE_MAX_PHOTOS = 3
    THOROUGH_MODE_MAX_PHOTOS = 15
    
    # Confidence thresholds
    VERIFIED_THRESHOLD = 70
    APPROXIMATE_THRESHOLD = 50

# Create config instance
config = Config()