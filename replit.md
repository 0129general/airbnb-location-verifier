# Airbnb Location Verifier

## Overview

This is an advanced Flask web application that helps users verify the exact location of Airbnb properties using multi-signal data extraction. The application combines OCR-based image analysis, NLP text extraction, web scraping, real estate database cross-referencing, and geocoding to provide highly accurate address verification with confidence scoring. It implements a sophisticated weighted scoring algorithm that achieves proper differentiation between Verified (≥70%), Approximate (50-69%), and Low Confidence (<50%) results.

**Latest Updates (October 27, 2025)**:
- **Deep Analysis Feature Complete**: 
  - Implemented comprehensive Deep Analysis mode with 30-45 second processing window
  - Added enforceable timing controls with deadline budget system and stage-specific timeouts
  - Enhanced vision analyzer to extract landmarks, building features, and environmental context from up to 15 photos
  - Integrated Street View metadata extraction and optional real estate cross-referencing
  - Multi-signal scoring now combines OCR, NLP, visual analysis, and geocoding with dynamic weight adjustment
  - Added comprehensive evidence breakdown tracking with processing stages and timing details
  - New ranked address candidates system with confidence scoring for each candidate
  - **Important**: Background worker must be started separately using `./start_background_worker.sh` for Deep Analysis to function
- **Previous Updates (October 26, 2025)**:
  - **Quick Verification System Overhaul**: Fixed critical UnboundLocalError, enhanced location accuracy, improved confidence scoring
  - **Modern UI Redesign**: New quick verification page, redesigned results page with circular confidence meter
  - **Performance Optimizations**: Thread-safe caching system, async vision analysis, 34.4% performance improvement
  - **Bug Fixes**: Fixed coordinate handling, resolved SQLAlchemy warnings, added proper timeout handling

**New Queue-Based System**: The application now features a two-tier verification system optimized for hourly data entry workers:
- **Quick Verify Mode** (Default): 5-10 second analysis for immediate results
- **Deep Analysis Queue**: Background processing for thorough verification with rate limiting to avoid Airbnb blocking

**New Monitoring & Observability**: Complete Prometheus-based monitoring with real-time dashboard:
- **Metrics Collection**: Comprehensive tracking of verifications, API calls, system resources, and background jobs
- **Real-Time Dashboard**: Beautiful monitoring interface at `/monitoring` with charts and alerts
- **Prometheus Integration**: Standard `/metrics` endpoint for external monitoring tools

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask web framework with Python
- **Web Scraping**: Custom scraper using BeautifulSoup and Trafilatura for extracting Airbnb listing data
- **Geocoding**: Dual approach using Nominatim (fallback) and Google Maps API (when available)
- **AI Integration**: OpenAI GPT-4 Vision for OCR-based address extraction from property photos
- **NLP Extraction**: Regex-based extraction of street names, HOA names, and POIs from descriptions
- **Multi-Signal Scoring**: Weighted algorithm combining proximity (40%), location type (20%), house number (20%), street name (10%), HOA/POI (5%), visual features (5%)
- **Database**: SQLAlchemy ORM with session-based result storage and caching
- **Resilience Patterns**: Circuit breaker pattern (5 failure threshold, 60s recovery), automatic retry mechanism (3 attempts with exponential backoff), and fallback strategies for all external APIs

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 for responsive UI
- **CSS Framework**: Bootstrap 5 with custom styling using CSS variables
- **JavaScript**: Vanilla JavaScript for form handling, map initialization, and clipboard functionality
- **Maps**: Leaflet.js for interactive maps with optional Google Maps API integration

### Data Processing
- **Single Processing**: Real-time scraping and analysis of individual Airbnb URLs
- **Bulk Processing**: Excel file upload and batch processing capabilities
- **Safe Mode**: Fallback processing mode that bypasses actual Airbnb scraping for testing

## Key Components

### Core Modules
1. **app.py**: Main Flask application with routing, multi-signal processing, and session management
2. **scraper.py**: Web scraping logic for extracting Airbnb location data and NLP integration
3. **vision_analyzer.py**: OpenAI GPT-4 Vision integration for OCR-based address extraction from photos
4. **google_vision_ocr.py**: Google Cloud Vision API for state-of-the-art OCR with superior house number detection
5. **tesseract_ocr.py**: Tesseract OCR for enhanced house number detection (fallback when Google Vision unavailable)
6. **nlp_extractor.py**: NLP extraction of street names, HOA names, and POIs from descriptions
7. **multi_signal_scorer.py**: Weighted scoring algorithm with dynamic weight adjustment based on available services
8. **address_normalizer.py**: Address parsing, normalization, and fuzzy matching
9. **real_estate_searcher.py**: Cross-referencing with Zillow/Realtor.com with rate limiting and caching
10. **ai_helpers.py**: OpenAI integration for neighborhood insights
11. **streetview_matcher.py**: Street View visual comparison module (disabled due to timeout constraints)
12. **models.py**: SQLAlchemy database models with session-based result storage
13. **main.py**: Application entry point

### Template Structure
- **layout.html**: Base template with navigation and common elements
- **index.html**: Homepage with single URL input form
- **result.html**: Location verification results display
- **bulk_upload.html**: Bulk processing interface
- **jobs.html**: Batch job status monitoring

### Static Assets
- **custom.css**: Custom styling with consistent color scheme and branding
- **main.js**: Frontend interactivity and map initialization

## Data Flow

1. **Single URL Processing**:
   - User submits Airbnb URL through web form
   - Scraper extracts location data from listing page
   - Geocoding services convert addresses to coordinates
   - Optional AI analysis of property descriptions
   - Results displayed with interactive map

2. **Bulk Processing**:
   - User uploads Excel file with multiple URLs
   - System processes URLs in batches
   - Progress tracking for large jobs
   - Results exported back to Excel format

3. **Location Verification**:
   - Extract coordinates from Airbnb's embedded data
   - Cross-reference with geocoded addresses
   - Display verification status and confidence levels

## External Dependencies

### Required Services
- **Airbnb**: Target platform for scraping listing data
- **Nominatim (OpenStreetMap)**: Free geocoding service (fallback)

### Optional Services
- **Google Maps API**: Enhanced geocoding and mapping features
- **Google Cloud Vision API**: State-of-the-art OCR for superior house number detection
- **OpenAI API**: AI-powered property description analysis
- **Google Street View API**: Integrated street-level property verification with interactive 360° view and static preview

### Python Packages
- **Flask**: Web framework
- **BeautifulSoup4**: HTML parsing
- **Trafilatura**: Content extraction
- **Geopy**: Geocoding utilities
- **GoogleMaps**: Google Maps API client
- **OpenAI**: AI integration
- **SQLAlchemy**: Database ORM

## Deployment Strategy

### Environment Configuration
- **Development**: Local Flask development server
- **Environment Variables**: API keys and configuration through environment variables
- **Safe Mode**: Testing mode that bypasses external API calls

### Key Environment Variables
- `OPENAI_API_KEY`: Enables AI features
- `GOOGLE_MAPS_API_KEY`: Enables Google Maps integration
- `SESSION_SECRET`: Flask session security
- `SAFE_MODE`: Testing mode toggle

### Architecture Decisions

1. **Dual Geocoding Approach**: Uses free Nominatim service as fallback with Google Maps as premium option, ensuring functionality regardless of API availability.

2. **Optional AI Integration**: OpenAI features are gracefully disabled when API key is unavailable, maintaining core functionality.

3. **Safe Mode Implementation**: Allows testing and development without external API dependencies or rate limiting concerns.

4. **Bootstrap + Custom CSS**: Provides professional UI with consistent branding while maintaining responsive design.

5. **Modular Structure**: Separates concerns between scraping, AI processing, and web interface for maintainability.

6. **Street View Visual Matching** (Currently Disabled): Implemented module to compare Airbnb photos with Google Street View images to distinguish between similar addresses (e.g., 141 vs 151 Pine St). Currently disabled due to 30-second timeout constraints but ready for async/background processing implementation.
7. **Tesseract OCR Enhancement**: Analyzes first 15 photos for house numbers (matching GPT-4 Vision coverage), with optimized preprocessing and region detection for comprehensive number detection