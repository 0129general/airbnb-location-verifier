import os
import sys
import logging
import time
import threading
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, make_response, Response, g
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import csv
import json
from io import StringIO
from datetime import datetime, timedelta
from sqlalchemy import func
from functools import wraps
from contextlib import contextmanager

# Import from new modular structure
from extraction.scraper import get_airbnb_location_data, get_google_search_results, get_street_view_metadata
from scoring.real_estate_searcher import RealEstateSearcher
from core.models import db, VerificationResult, JobQueue
from scoring.multi_signal_scorer import select_best_address

# Import Prometheus metrics
from core.metrics import (
    # Request/Response metrics
    http_requests_total, http_request_duration,
    
    # Verification metrics
    verification_counter, verification_time, confidence_gauge, address_source_counter,
    
    # Background job metrics
    active_jobs, job_completions, job_processing_time,
    
    # External API metrics
    external_api_calls, external_api_latency, api_errors,
    
    # OCR/Vision metrics
    ocr_operations, ocr_processing_time, ocr_confidence, images_processed,
    
    # Database metrics
    db_operations, db_query_time, db_connections,
    
    # System metrics
    cpu_usage, memory_usage, memory_bytes, disk_usage, app_uptime,
    app_info, active_sessions, sessions_cleaned,
    
    # Helper functions
    track_request_metrics, track_verification_metrics, track_external_api_call,
    track_ocr_operation, update_system_metrics,
    
    # Generate metrics
    generate_latest
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if real estate search is enabled (default: False)
ENABLE_REAL_ESTATE_SEARCH = os.environ.get("ENABLE_REAL_ESTATE_SEARCH", "false").lower() in ["true", "1", "yes", "on"]
if ENABLE_REAL_ESTATE_SEARCH:
    logger.info("Real estate search is ENABLED")
else:
    logger.info("Real estate search is DISABLED - Using Advanced OCR & NLP Analysis")

# Import AI helper functions if OpenAI API key is available
ai_neighborhood_insights = None
ai_property_analyzer = None
vision_analyzer = None
if os.environ.get("OPENAI_API_KEY"):
    try:
        from ai.ai_helpers import generate_neighborhood_insights, analyze_property_description
        ai_neighborhood_insights = generate_neighborhood_insights
        ai_property_analyzer = analyze_property_description
        logger.info("OpenAI API key found - AI neighborhood features enabled")
    except ImportError:
        logger.warning("Could not import AI helpers despite API key being present")
    
    try:
        from ocr.vision_analyzer import extract_address_from_visual_context
        vision_analyzer = extract_address_from_visual_context
        logger.info("Vision analysis features enabled")
    except ImportError as e:
        logger.warning(f"Could not import vision analyzer: {e}")
else:
    logger.info("OpenAI API key not found - AI features disabled")

# Create Flask app with proper template and static paths
# Since app.py is in src/core/, templates are at ../../templates
template_dir = Path(__file__).resolve().parent.parent.parent / 'templates'
static_dir = Path(__file__).resolve().parent.parent.parent / 'static'

app = Flask(__name__, 
           template_folder=str(template_dir),
           static_folder=str(static_dir))
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///verification_results.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
db.init_app(app)

# Background cleanup thread
def background_cleanup():
    """Background thread for cleaning up expired sessions."""
    with app.app_context():
        while True:
            time.sleep(3600)  # Run every hour
            try:
                count = VerificationResult.cleanup_expired(expiry_hours=24)
                if count > 0:
                    logger.info(f"Background cleanup: removed {count} expired sessions")
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

# Request/Response Middleware for Metrics
@app.before_request
def before_request():
    """Track request start time for latency measurement."""
    g.start_time = time.time()

@app.after_request
def after_request(response):
    """Track HTTP request metrics after each request."""
    # Skip metrics endpoint itself to avoid recursion
    if request.path == '/metrics':
        return response
    
    # Calculate request duration
    request_duration = time.time() - getattr(g, 'start_time', time.time())
    
    # Track metrics
    track_request_metrics(
        method=request.method,
        endpoint=request.endpoint or request.path,
        status_code=response.status_code
    )
    
    # Track duration
    http_request_duration.labels(
        method=request.method,
        endpoint=request.endpoint or request.path
    ).observe(request_duration)
    
    return response

# Metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    # Update system metrics before serving
    update_system_metrics()
    
    # Update active sessions count
    active_count = VerificationResult.query.filter(
        VerificationResult.accessed_at > datetime.utcnow() - timedelta(hours=24)
    ).count()
    active_sessions.set(active_count)
    
    # Update cache metrics
    from utils.location_cache import update_cache_metrics, get_cache
    update_cache_metrics()
    
    # Log cache stats for monitoring
    cache_stats = get_cache().get_stats()
    logger.info(f"Cache stats: hit_rate={cache_stats['hit_rate']:.1f}%, hits={cache_stats['hits']}, "
                f"misses={cache_stats['misses']}, size={cache_stats['cache_size']}")
    
    # Generate and return metrics
    return Response(generate_latest(), mimetype="text/plain")

# Initialize database and background thread
def initialize_app():
    """Initialize database and start background threads"""
    with app.app_context():
        db.create_all()
        
        # Initialize app info metrics
        app_info.info({
            'version': '1.0.0',
            'environment': os.environ.get('ENVIRONMENT', 'development'),
            'apify_enabled': str(os.environ.get("ENABLE_APIFY", "false")),
            'ai_enabled': str(bool(os.environ.get("OPENAI_API_KEY"))),
            'real_estate_enabled': str(ENABLE_REAL_ESTATE_SEARCH)
        })
        
        # Start background cleanup thread instead of cleaning on startup
        cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info("Background cleanup thread started - runs every hour")

# Only initialize if running directly (not when imported by gunicorn)
if __name__ == "__main__":
    initialize_app()

@app.route("/", methods=["GET"])
def index():
    """Render the homepage with the URL input form."""
    return render_template("index.html")

def process_in_background(session_id, continue_from_stage='initial'):
    """
    Process the extraction in a background thread with staged analysis.
    Implements unlimited timeout for comprehensive analysis.
    
    Args:
        session_id: The session ID to process
        continue_from_stage: Stage to continue from ('initial', 'quick', 'detailed')
    """
    import json
    from datetime import datetime
    
    # Track active background job
    active_jobs.labels(job_type='verification_background').inc()
    job_start_time = time.time()
    
    with app.app_context():
        # Get the verification result
        verification_result = VerificationResult.get_by_session(session_id)
        if not verification_result:
            logger.error(f"Session {session_id} not found for background processing")
            return
        
        try:
            # Update status to processing
            verification_result.processing_status = 'processing'
            verification_result.processing_started_at = datetime.utcnow()
            db.session.commit()
            
            # Get the data from the result
            data = verification_result.data
            url = data.get('original_url')
            location_data = data.get('location_data')
            
            # Check if we have already completed quick analysis
            has_quick_results = data.get('quick_analysis_completed', False)
            
            # Perform the full analysis without timeout restrictions
            logger.info(f"Background processing starting for session {session_id} from stage: {continue_from_stage}")
            
            # Visual analysis (can take as long as needed)
            visual_analysis_results = None
            if vision_analyzer and location_data.get('photos'):
                try:
                    # Update progress
                    verification_result.processing_progress = json.dumps({'stage': 'vision_analysis', 'message': 'Analyzing property photos with AI vision...'})
                    db.session.commit()
                    
                    # Determine photo count based on analysis mode
                    analysis_mode = verification_result.data.get('analysis_mode', 'quick')
                    if analysis_mode == 'thorough':
                        max_photos_to_analyze = 15  # Full analysis with all photos
                        logger.info(f"[Background] THOROUGH MODE: Analyzing all {max_photos_to_analyze} photos")
                    else:
                        max_photos_to_analyze = 5  # Quick mode for faster results
                        logger.info(f"[Background] QUICK MODE: Analyzing only {max_photos_to_analyze} photos for speed")
                    
                    photos_to_analyze = location_data['photos'][:max_photos_to_analyze]
                    logger.info(f"[Background] Performing visual analysis on {len(photos_to_analyze)} photos")
                    
                    coords = None
                    if location_data.get('latitude') and location_data.get('longitude'):
                        coords = (location_data['latitude'], location_data['longitude'])
                    
                    start_time = time.time()
                    visual_analysis_results = vision_analyzer(photos_to_analyze, coords)
                    elapsed = time.time() - start_time
                    logger.info(f"[Background] Vision analysis completed in {elapsed:.2f} seconds")
                    
                    # Update address if visual analysis found a better one
                    if visual_analysis_results and visual_analysis_results.get('suggested_address'):
                        confidence_val = visual_analysis_results.get('final_address_confidence', 0)
                        if isinstance(confidence_val, (int, float)) and confidence_val > 70:
                            new_address = visual_analysis_results['suggested_address']
                            current_address = location_data.get('address', '')
                            
                            if new_address and isinstance(new_address, str) and (not current_address or 
                                               (isinstance(current_address, str) and (len(new_address.split(',')) > len(current_address.split(',')) or
                                               (len(new_address) > len(current_address) and 'coastal' not in new_address.lower())))):
                                location_data['original_address'] = location_data.get('address', 'Unknown')
                                location_data['address'] = new_address
                                location_data['address_source'] = 'visual_analysis'
                                logger.info(f"[Background] Updated address from visual analysis: {new_address}")
                    
                    # Add visual clues to location data
                    if visual_analysis_results:
                        location_data['visual_clues'] = visual_analysis_results.get('visual_analysis', {})
                        
                except Exception as vision_error:
                    logger.error(f"[Background] Error in visual analysis: {str(vision_error)}")
                    visual_analysis_results = None
            
            # Street View metadata (also without timeout)
            street_view_metadata = None
            if location_data.get('latitude') and location_data.get('longitude'):
                try:
                    # Update progress
                    verification_result.processing_progress = json.dumps({'stage': 'street_view', 'message': 'Fetching Street View data...'})
                    db.session.commit()
                    
                    logger.info(f"[Background] Fetching Street View metadata")
                    street_view_metadata = get_street_view_metadata(
                        location_data['latitude'],
                        location_data['longitude']
                    )
                    location_data['street_view_metadata'] = street_view_metadata
                    
                    if street_view_metadata.get('available'):
                        logger.info(f"[Background] Street View available - Panorama ID: {street_view_metadata.get('panorama_id')}")
                        
                except Exception as sv_error:
                    logger.error(f"[Background] Error getting Street View metadata: {str(sv_error)}")
                    street_view_metadata = {
                        'available': False,
                        'status': 'ERROR',
                        'error_message': str(sv_error)
                    }
                    location_data['street_view_metadata'] = street_view_metadata
            
            # Multi-signal scoring
            multi_signal_result = None
            try:
                # Update progress
                verification_result.processing_progress = json.dumps({'stage': 'scoring', 'message': 'Performing multi-signal scoring...'})
                db.session.commit()
                
                logger.info(f"[Background] Starting multi-signal scoring")
                multi_signal_result = select_best_address(
                    scraped_data=location_data,
                    ocr_data=visual_analysis_results,
                    nlp_data=location_data.get('nlp_extraction'),
                    vision_data=visual_analysis_results,
                    airbnb_photos=location_data.get('photos', []),
                    real_estate_enabled=False
                )
                
                if multi_signal_result and multi_signal_result.get('selected_address'):
                    original_address = location_data.get('address', 'Unknown')
                    selected_address = multi_signal_result['selected_address']
                    confidence_score = multi_signal_result.get('confidence_score', 0)
                    
                    if confidence_score > 30 and selected_address != original_address:
                        location_data['original_scraped_address'] = original_address
                        location_data['address'] = selected_address
                        location_data['address_source'] = 'multi_signal_scoring'
                        location_data['address_confidence'] = confidence_score
                        location_data['confidence_level'] = multi_signal_result.get('confidence_level', 'Unknown')
                        logger.info(f"[Background] Updated address via multi-signal scoring: {selected_address} (confidence: {confidence_score:.1f}%)")
                    
                    location_data['multi_signal_scoring'] = {
                        'selected_address': multi_signal_result.get('selected_address'),
                        'confidence_score': multi_signal_result.get('confidence_score'),
                        'confidence_level': multi_signal_result.get('confidence_level'),
                        'evidence': multi_signal_result.get('evidence', {}),
                        'contributions': multi_signal_result.get('contributions', {}),
                        'component_scores': multi_signal_result.get('component_scores', {}),
                        'all_candidates': multi_signal_result.get('all_candidates', [])[:5]
                    }
                    
            except Exception as scoring_error:
                logger.error(f"[Background] Error in multi-signal scoring: {str(scoring_error)}")
            
            # Update the verification result with all the processed data
            result_data = {
                'location_data': location_data,
                'original_url': url,
                'google_maps_api_key': os.environ.get('GOOGLE_MAPS_API_KEY', ''),
                'search_results': {},
                'real_estate_matches': {},
                'visual_analysis_results': visual_analysis_results,
                'multi_signal_result': multi_signal_result,
                'ai_enabled': bool(ai_neighborhood_insights),
                'real_estate_enabled': False
            }
            
            verification_result.extracted_data = json.dumps(result_data, default=str)
            verification_result.processing_status = 'completed'
            verification_result.processing_completed_at = datetime.utcnow()
            verification_result.processing_progress = json.dumps({'stage': 'completed', 'message': 'Processing completed successfully!'})
            db.session.commit()
            
            logger.info(f"[Background] Processing completed for session {session_id}")
            
            # Track job completion metrics
            job_duration = time.time() - job_start_time
            job_processing_time.labels(job_type='verification_background').observe(job_duration)
            job_completions.labels(job_type='verification_background', status='success').inc()
            active_jobs.labels(job_type='verification_background').dec()
            logger.info(f"[Background] Job completed in {job_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"[Background] Error processing session {session_id}: {str(e)}")
            verification_result.processing_status = 'error'
            verification_result.processing_error = str(e)
            verification_result.processing_completed_at = datetime.utcnow()
            db.session.commit()
            
            # Track job failure metrics
            job_duration = time.time() - job_start_time
            job_processing_time.labels(job_type='verification_background').observe(job_duration)
            job_completions.labels(job_type='verification_background', status='failure').inc()
            active_jobs.labels(job_type='verification_background').dec()
            api_errors.labels(service='background_processing', error_type=type(e).__name__).inc()

def process_quick_verify_background(session_id, location_data):
    """
    Background processing for quick verify results to add Street View metadata.
    """
    # Track active background job
    active_jobs.labels(job_type='quick_verify_background').inc()
    job_start_time = time.time()
    
    with app.app_context():
        verification_result = None  # Initialize to None
        try:
            logger.info(f"[Quick Background] Starting background processing for session {session_id}")
            
            # Get the verification result
            verification_result = VerificationResult.get_by_session(session_id)
            if not verification_result:
                logger.error(f"[Quick Background] Session not found: {session_id}")
                return
            
            # Get current data
            result_data = verification_result.data
            
            # Fetch Street View metadata if we have coordinates
            if location_data.get('latitude') and location_data.get('longitude'):
                try:
                    logger.info(f"[Quick Background] Fetching Street View metadata")
                    street_view_metadata = get_street_view_metadata(
                        location_data['latitude'],
                        location_data['longitude']
                    )
                    location_data['street_view_metadata'] = street_view_metadata
                    
                    if street_view_metadata.get('available'):
                        logger.info(f"[Quick Background] Street View available - Panorama ID: {street_view_metadata.get('panorama_id')}")
                    else:
                        logger.info(f"[Quick Background] Street View not available at this location")
                        
                except Exception as sv_error:
                    logger.error(f"[Quick Background] Error getting Street View metadata: {str(sv_error)}")
                    street_view_metadata = {
                        'available': False,
                        'status': 'ERROR',
                        'error_message': str(sv_error)
                    }
                    location_data['street_view_metadata'] = street_view_metadata
            else:
                logger.info(f"[Quick Background] No coordinates available for Street View")
            
            # Update the verification result with the enhanced location data
            result_data['location_data'] = location_data
            verification_result.data = result_data
            verification_result.processing_status = 'completed'
            verification_result.processing_completed_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"[Quick Background] Completed background processing for session {session_id}")
            
            # Track job completion metrics
            job_duration = time.time() - job_start_time
            job_processing_time.labels(job_type='quick_verify_background').observe(job_duration)
            job_completions.labels(job_type='quick_verify_background', status='success').inc()
            active_jobs.labels(job_type='quick_verify_background').dec()
            
        except Exception as e:
            logger.error(f"[Quick Background] Error in background processing: {str(e)}")
            try:
                # Only update if we have a verification_result
                if verification_result is not None:
                    verification_result.processing_status = 'completed'  # Still mark as completed even if Street View fails
                    verification_result.processing_completed_at = datetime.utcnow()
                    db.session.commit()
            except:
                pass
            
            # Track job failure metrics
            job_duration = time.time() - job_start_time
            job_processing_time.labels(job_type='quick_verify_background').observe(job_duration)
            job_completions.labels(job_type='quick_verify_background', status='partial').inc()
            active_jobs.labels(job_type='quick_verify_background').dec()

def quick_verify_extraction(url, timeout=10):
    """
    Perform a quick verification extraction with enhanced validation and accuracy.
    Returns in 10 seconds maximum with calibrated confidence level.
    Now with caching and async vision analysis for improved performance.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    try:
        start_time = time.time()
        
        # Try to get basic location data with timeout (will use cache automatically)
        location_data = get_airbnb_location_data(url)
        
        # Log if data came from cache
        if location_data and location_data.get('_from_cache'):
            logger.info(f"Quick verify using CACHED data for: {url[:50]}...")
            # Remove cache metadata from result
            location_data.pop('_from_cache', None)
            location_data.pop('_cache_timestamp', None)
        
        # Check if the listing is no longer available (404)
        if location_data and location_data.get("status_code") == 404:
            logger.info(f"Listing is no longer available: {url}")
            return {
                'success': False,
                'confidence': 0,
                'status': 'not_found',
                'message': location_data.get('message', 'This Airbnb listing is no longer active or has been removed.'),
                'location_data': location_data
            }
        
        if not location_data or location_data.get("error"):
            return {
                'success': False,
                'confidence': 0,
                'status': 'inconclusive',
                'message': 'Could not extract basic location data',
                'location_data': {}
            }
        
        # Enhanced validation and normalization
        # Validate coordinates are within valid ranges
        lat = location_data.get('latitude')
        lng = location_data.get('longitude')
        if lat and lng:
            try:
                lat = float(lat)
                lng = float(lng)
                if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                    logger.warning(f"Invalid coordinates: {lat}, {lng}")
                    location_data['latitude'] = None
                    location_data['longitude'] = None
            except (ValueError, TypeError):
                location_data['latitude'] = None
                location_data['longitude'] = None
        
        # Calculate enhanced confidence with dynamic weights
        confidence = 0
        confidence_factors = []
        evidence_quality = {}
        
        # Address quality assessment (max 35 points)
        if location_data.get('address'):
            address = str(location_data['address']).strip()
            # Normalize address for better consistency
            address_parts = [p.strip() for p in address.split(',') if p.strip()]
            
            # Check for street number and name
            has_street_number = any(char.isdigit() for char in address_parts[0]) if address_parts else False
            
            if len(address_parts) >= 4:  # Full address with country
                confidence += 35
                confidence_factors.append('complete_address')
                evidence_quality['address'] = 'high'
            elif len(address_parts) >= 3:  # City, state/region
                confidence += 25
                confidence_factors.append('detailed_address')
                evidence_quality['address'] = 'medium'
            elif len(address_parts) >= 2:
                confidence += 15
                confidence_factors.append('partial_address')
                evidence_quality['address'] = 'low'
            else:
                confidence += 5
                confidence_factors.append('minimal_address')
                evidence_quality['address'] = 'minimal'
            
            # Bonus for street number
            if has_street_number:
                confidence += 5
                confidence_factors.append('has_street_number')
        
        # Coordinate validation (max 25 points)
        if location_data.get('latitude') and location_data.get('longitude'):
            confidence += 25
            confidence_factors.append('valid_coordinates')
            evidence_quality['coordinates'] = 'verified'
        
        # Quick visual analysis (max 20 points) - enhanced with async execution
        visual_confidence_added = False
        vision_future = None
        executor = None
        coords = None  # Define coords outside try block to prevent UnboundLocalError
        
        if vision_analyzer and location_data.get('photos'):
            remaining_time = max(2, timeout - (time.time() - start_time))
            if remaining_time > 2:
                try:
                    # Analyze first 3 photos for better accuracy
                    photos_to_check = location_data['photos'][:3]
                    logger.info(f"Quick verify: Enhanced ASYNC analysis of {len(photos_to_check)} photos")
                    
                    if location_data.get('latitude') and location_data.get('longitude'):
                        coords = (location_data['latitude'], location_data['longitude'])
                    
                    # Execute vision analysis asynchronously with timeout
                    executor = ThreadPoolExecutor(max_workers=1)
                    vision_future = executor.submit(vision_analyzer, photos_to_check, coords)
                    
                    # Wait for result with timeout
                    try:
                        # Use remaining time minus 1 second for safety
                        async_timeout = max(1, remaining_time - 1)
                        vision_result = vision_future.result(timeout=async_timeout)
                    except FutureTimeoutError:
                        logger.warning(f"Vision analysis timed out after {async_timeout:.1f}s")
                        vision_result = None
                    
                    if vision_result and vision_result.get('suggested_address'):
                        # Compare with existing address
                        vision_confidence = vision_result.get('final_address_confidence', 0)
                        if vision_confidence > 60:
                            confidence += 20
                            confidence_factors.append('vision_verified')
                            evidence_quality['vision'] = 'strong'
                            visual_confidence_added = True
                            
                            # Update address if vision found better one
                            if vision_confidence > 75:
                                vision_address = vision_result['suggested_address']
                                current_address = location_data.get('address', '')
                                if len(vision_address.split(',')) > len(str(current_address).split(',')):
                                    location_data['address'] = vision_address
                                    location_data['address_source'] = 'quick_vision'
                                    confidence_factors.append('vision_enhanced_address')
                        elif vision_confidence > 40:
                            confidence += 10
                            confidence_factors.append('vision_partial')
                            evidence_quality['vision'] = 'weak'
                            visual_confidence_added = True
                    
                    # Store vision metadata
                    if vision_result:
                        location_data['quick_vision_analysis'] = {
                            'confidence': vision_result.get('final_address_confidence', 0),
                            'signals_found': len(vision_result.get('visual_analysis', {}).get('text_found', []))
                        }
                        
                except Exception as e:
                    logger.warning(f"Quick vision check failed: {e}")
                finally:
                    # Clean up executor
                    if executor:
                        executor.shutdown(wait=False)
            
            # Fallback if vision didn't work
            if not visual_confidence_added and coords:
                confidence += 5
                confidence_factors.append('photos_available')
        
        # NLP/description analysis (max 10 points)
        if location_data.get('nlp_extraction'):
            nlp_data = location_data['nlp_extraction']
            if nlp_data.get('overall_confidence', 0) > 50:
                confidence += 10
                confidence_factors.append('nlp_verified')
                evidence_quality['nlp'] = 'present'
        
        # Proximity and neighborhood data (max 10 points)
        if location_data.get('neighborhood'):
            confidence += 5
            confidence_factors.append('has_neighborhood')
        
        if location_data.get('verification', {}).get('proximity_ok'):
            confidence += 5
            confidence_factors.append('proximity_verified')
        
        # Apply calibration and minimum requirements
        # Require at least 2 evidence sources for high confidence
        evidence_count = len([v for v in evidence_quality.values() if v])
        
        if evidence_count < 2 and confidence > 60:
            # Downgrade confidence if only one signal
            confidence = min(confidence, 55)
            confidence_factors.append('single_source_limitation')
        
        # Maximum confidence without street view or vision is 85
        if 'vision_verified' not in confidence_factors and confidence > 85:
            confidence = 85
        
        # Determine calibrated status
        if confidence >= 75 and evidence_count >= 2:
            status = 'verified'
        elif confidence >= 50:
            status = 'approximate'
        elif confidence >= 30:
            status = 'low_confidence'
        else:
            status = 'inconclusive'
        
        elapsed = time.time() - start_time
        logger.info(f"Enhanced quick verify completed in {elapsed:.2f}s with confidence {confidence}% ({evidence_count} sources)")
        
        return {
            'success': True,
            'confidence': confidence,
            'confidence_factors': confidence_factors,
            'evidence_quality': evidence_quality,
            'evidence_count': evidence_count,
            'status': status,
            'location_data': location_data,
            'processing_time': elapsed,
            'mode': 'quick_verify_enhanced'
        }
        
    except Exception as e:
        logger.error(f"Quick verify error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'confidence': 0,
            'status': 'error',
            'message': str(e),
            'location_data': {}
        }


@app.route("/api/processing-status/<session_id>", methods=["GET"])
def get_processing_status(session_id):
    """
    API endpoint to check processing status and get incremental results.
    Returns JSON with current status and any available results.
    """
    verification_result = VerificationResult.get_by_session(session_id)
    if not verification_result:
        return jsonify({"error": "Session not found"}), 404
    
    # Get current processing status
    status = verification_result.processing_status or 'pending'
    progress = json.loads(verification_result.processing_progress or '{}')
    
    # Prepare response based on status
    response_data = {
        "session_id": session_id,
        "status": status,
        "progress": progress,
        "processing_started_at": verification_result.processing_started_at.isoformat() if verification_result.processing_started_at else None,
        "processing_completed_at": verification_result.processing_completed_at.isoformat() if verification_result.processing_completed_at else None
    }
    
    # Include partial results if available
    if verification_result.data:
        data = verification_result.data
        response_data["has_quick_results"] = data.get('quick_analysis_completed', False)
        response_data["has_detailed_results"] = data.get('detailed_analysis_completed', False)
        response_data["has_final_results"] = status == 'completed'
        
        # Include address if available
        if data.get('location_data', {}).get('address'):
            response_data["current_address"] = data['location_data']['address']
            response_data["confidence_score"] = data['location_data'].get('address_confidence', 0)
            
    # Include any extracted data if processing is complete
    if status == 'completed' and verification_result.extracted_data:
        try:
            extracted = json.loads(verification_result.extracted_data)
            response_data["visual_analysis_completed"] = bool(extracted.get('visual_analysis_results'))
            response_data["multi_signal_completed"] = bool(extracted.get('multi_signal_result'))
        except:
            pass
            
    return jsonify(response_data)

@app.route("/extract", methods=["GET", "POST"])
def extract():
    """
    Extract location data from the provided Airbnb URL.
    Implements PROGRESSIVE LOADING:
    - Returns basic data immediately (within 5 seconds)
    - Continues detailed analysis in background (30 seconds)
    - Full analysis runs unlimited in background
    """
    # Handle GET requests by redirecting to homepage
    if request.method == "GET":
        flash("Please enter an Airbnb URL on the homepage first.", "info")
        return redirect(url_for("index"))
    
    # Track extraction start time for metrics
    extraction_start_time = time.time()
    
    url = request.form.get("airbnb_url", "").strip()
    analysis_mode = request.form.get("analysis_mode", "quick")
    
    # Comprehensive URL validation
    if not url:
        flash("Please enter an Airbnb listing URL", "danger")
        return redirect(url_for("index"))
    
    # Validate URL format
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.error(f"Invalid URL format: {e}")
        flash("Invalid URL format. Please enter a valid web address.", "danger")
        return redirect(url_for("index"))
    
    # Check if it's an Airbnb domain
    if not parsed.netloc:
        flash("Invalid URL. Please include the complete Airbnb URL including https://", "danger")
        return redirect(url_for("index"))
    
    # List of valid Airbnb domains
    valid_domains = [
        'airbnb.com', 'airbnb.co.uk', 'airbnb.ca', 'airbnb.com.au',
        'airbnb.de', 'airbnb.fr', 'airbnb.es', 'airbnb.it', 'airbnb.jp',
        'airbnb.com.br', 'airbnb.mx', 'airbnb.nl', 'airbnb.pt', 'airbnb.se',
        'airbnb.dk', 'airbnb.no', 'airbnb.fi', 'airbnb.ie', 'airbnb.at',
        'airbnb.ch', 'airbnb.be', 'airbnb.gr', 'airbnb.pl', 'airbnb.ru',
        'airbnb.co.kr', 'airbnb.com.sg', 'airbnb.co.in', 'airbnb.co.nz',
        'airbnb.com.ar', 'airbnb.cl', 'airbnb.com.co', 'airbnb.com.pe'
    ]
    
    # Check if domain is valid (including subdomains like www. or m.)
    domain_valid = any(domain in parsed.netloc for domain in valid_domains)
    
    if not domain_valid:
        flash(f"This doesn't appear to be an Airbnb URL. Please use a URL from airbnb.com or other Airbnb domains.", "danger")
        return redirect(url_for("index"))
    
    # Check if it's a listing URL (contains /rooms/)
    if '/rooms/' not in parsed.path:
        flash("Please provide a specific Airbnb listing URL (should contain '/rooms/' in the path)", "warning")
        return redirect(url_for("index"))
    
    # Clean up URL - Keep only meaningful query params
    allowed_params = {"adults", "children", "check_in", "check_out", "rooms", "guests"}
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k in allowed_params]
    cleaned = parsed._replace(fragment="", query=urlencode(kept, doseq=True))
    url = urlunparse(cleaned)
    
    try:
        # Check if this is a quick verify request
        if analysis_mode == 'quick':
            # Perform quick verification with aggressive timeouts
            logger.info("Performing Quick Verify mode analysis")
            
            # Track quick verification
            with verification_time.labels(stage='quick_verify').time():
                quick_result = quick_verify_extraction(url, timeout=10)  # 10 seconds max
            
            # Check if listing is not found (404)
            if quick_result.get('status') == 'not_found':
                logger.info(f"Listing not found (404): {url}")
                verification_counter.labels(status='not_found', mode='quick').inc()
                return render_template("listing_unavailable.html", airbnb_url=url)
            
            if quick_result['success']:
                # Track successful quick verification
                verification_counter.labels(status='success', mode='quick').inc()
                confidence_gauge.set(quick_result.get('confidence', 0))
                
                # Create a session for quick verify results
                verification_result = VerificationResult.create_session(url, quick_result['location_data'])
                verification_result.data = {
                    'original_url': url,
                    'location_data': quick_result['location_data'],
                    'confidence': quick_result['confidence'],
                    'confidence_factors': quick_result.get('confidence_factors', []),
                    'status': quick_result['status'],
                    'mode': 'quick_verify',
                    'processing_time': quick_result.get('processing_time', 0)
                }
                verification_result.processing_status = 'processing'  # Mark as processing initially
                verification_result.processing_started_at = datetime.utcnow()
                db.session.add(verification_result)
                db.session.commit()
                
                # Start background processing to add street view and other metadata
                logger.info(f"Starting background processing for quick verify session: {verification_result.session_id}")
                background_thread = threading.Thread(
                    target=process_quick_verify_background,
                    args=(verification_result.session_id, quick_result['location_data']),
                    daemon=True
                )
                background_thread.start()
                
                # Redirect to a quick results page
                return redirect(url_for("quick_results", session_id=verification_result.session_id))
            else:
                # Track failed quick verification
                verification_counter.labels(status='failure', mode='quick').inc()
                # If quick verify fails, offer to queue for deep analysis
                flash("Quick verification was unable to extract location data. You can queue this for deep analysis.", "warning")
                return render_template("queue_offer.html", url=url)
        
        # Original thorough analysis mode
        # Check for cached results first (within 24 hours)
        use_cache = True
        cache_duration = 24
        
        # Check if we should bypass cache (e.g., for testing or debugging)
        if request.args.get('fresh') == '1' or request.args.get('debug') == '1':
            use_cache = False
            logger.info("Cache bypass requested via URL parameter")
        
        if use_cache:
            cached_result = VerificationResult.get_cached(url, cache_hours=cache_duration)
            if cached_result:
                logger.info(f"Using cached result for URL: {url}")
                # Track cache hit
                db_operations.labels(operation='cache_hit', table='verification_results', status='success').inc()
                return redirect(url_for("results", session_id=cached_result.session_id))
            else:
                # Track cache miss
                db_operations.labels(operation='cache_miss', table='verification_results', status='success').inc()
        
        logger.debug(f"Attempting to extract location data from: {url}")
        location_data = get_airbnb_location_data(url)
        
        if not location_data:
            flash("Could not extract location data from this Airbnb listing. The listing may be unavailable or the page structure may have changed.", "warning")
            return redirect(url_for("index"))
        
        # Analyze enhanced description data if available (from Apify)
        ai_description_analysis = {}
        if location_data.get('apify_enhanced') and ai_property_analyzer:
            try:
                # Use the full combined description for comprehensive AI analysis
                full_description = location_data.get('full_description_text') or location_data.get('description_text', '')
                
                # If we have Apify data, include all the enhanced fields
                if location_data.get('apify_data'):
                    apify_data = location_data['apify_data']
                    # Combine all relevant text fields for analysis
                    combined_text = []
                    if apify_data.get('full_description'):
                        combined_text.append(apify_data['full_description'])
                    if apify_data.get('neighborhood_overview'):
                        combined_text.append(f"Neighborhood: {apify_data['neighborhood_overview']}")
                    if apify_data.get('transit_info'):
                        combined_text.append(f"Transit: {apify_data['transit_info']}")
                    if apify_data.get('getting_around'):
                        combined_text.append(f"Getting Around: {apify_data['getting_around']}")
                    if apify_data.get('space_description'):
                        combined_text.append(f"Space: {apify_data['space_description']}")
                    
                    if combined_text:
                        full_description = '\n\n'.join(combined_text)
                
                if full_description and len(full_description) > 50:
                    logger.info(f"Analyzing enhanced Apify description ({len(full_description)} chars) with AI...")
                    ai_description_analysis = ai_property_analyzer(full_description)
                    
                    # Log the AI analysis results
                    if ai_description_analysis:
                        logger.info(f"AI Analysis Results from Apify data:")
                        if ai_description_analysis.get('key_location_hints'):
                            logger.info(f"  - Location hints: {ai_description_analysis['key_location_hints'][:3]}")
                        if ai_description_analysis.get('landmark_mentions'):
                            logger.info(f"  - Landmarks: {ai_description_analysis['landmark_mentions'][:3]}")
                        if ai_description_analysis.get('distance_mentions'):
                            logger.info(f"  - Distances: {ai_description_analysis['distance_mentions'][:3]}")
                        
                        # Store the AI analysis in location_data for later use
                        location_data['ai_description_analysis'] = ai_description_analysis
                        
                        # Try to extract more specific address information from AI analysis
                        if ai_description_analysis.get('key_location_hints'):
                            for hint in ai_description_analysis['key_location_hints']:
                                if 'street' in hint.lower() or 'avenue' in hint.lower() or 'road' in hint.lower():
                                    logger.info(f"AI extracted potential street from description: {hint}")
                    
                    logger.info("Successfully completed AI analysis of enhanced Apify description")
            except Exception as e:
                logger.error(f"Error in AI description analysis: {str(e)}")
                # Continue with regular processing even if AI analysis fails
        elif not location_data.get('apify_enhanced'):
            logger.info("Apify enhanced data not available, using standard extraction")
        
        # Get Google search results for the address
        search_results = {}
        if location_data.get('address'):
            logger.debug(f"Fetching Google search results for address: {location_data['address']}")
            search_results = get_google_search_results(location_data['address'])
        
        # Use NLP-extracted data to verify and enhance address
        nlp_data = location_data.get('nlp_extraction', {})
        if nlp_data and nlp_data.get('overall_confidence', 0) > 0:
            logger.info(f"NLP extraction confidence: {nlp_data.get('overall_confidence')}%")
            
            # Log extracted entities for debugging
            if nlp_data.get('street_names'):
                logger.info(f"NLP extracted streets: {[s['street_name'] for s in nlp_data['street_names'][:3]]}")
            if nlp_data.get('hoa_names'):
                logger.info(f"NLP extracted HOAs: {[h['hoa_name'] for h in nlp_data['hoa_names'][:3]]}")
            if nlp_data.get('pois'):
                logger.info(f"NLP extracted POIs: {[p['poi_name'] for p in nlp_data['pois'][:3]]}")
            
            # Cross-reference NLP street names with the address
            if nlp_data.get('street_names') and location_data.get('address'):
                for street in nlp_data['street_names'][:3]:  # Check top 3 streets
                    if street['confidence'] > 70:
                        street_name = street['street_name']
                        # Check if this street is in the current address
                        if street_name.lower() not in location_data['address'].lower():
                            logger.warning(f"NLP found street '{street_name}' with confidence {street['confidence']}% but not in address: {location_data['address']}")
                            # This could indicate the address needs refinement
                            location_data['nlp_address_mismatch'] = True
        
        # PROGRESSIVE LOADING IMPLEMENTATION
        # Stage 1: Quick verification (5 seconds max) - return immediately
        visual_analysis_results = None
        quick_analysis_completed = False
        
        if vision_analyzer and location_data.get('photos'):
            try:
                logger.info("=== PROGRESSIVE LOADING: Starting quick verification (5s max) ===")
                photos_to_analyze = location_data['photos'][:2]  # Only 2 photos for quick analysis
                
                coords = None
                if location_data.get('latitude') and location_data.get('longitude'):
                    coords = (location_data['latitude'], location_data['longitude'])
                
                try:
                    start_time = time.time()
                    # Use the new staged analysis function
                    # This will automatically try 5s timeout, then 30s, then fallback
                    visual_analysis_results = vision_analyzer(photos_to_analyze, coords)
                    elapsed = time.time() - start_time
                    logger.info(f"Quick vision analysis completed in {elapsed:.2f} seconds")
                    quick_analysis_completed = True
                    
                    # Update address if visual analysis found a better one
                    if visual_analysis_results and visual_analysis_results.get('suggested_address'):
                        confidence_val = visual_analysis_results.get('final_address_confidence', 0)
                        if isinstance(confidence_val, (int, float)) and confidence_val > 70:
                            new_address = visual_analysis_results['suggested_address']
                            current_address = location_data.get('address', '')
                            
                            if new_address and isinstance(new_address, str) and (not current_address or 
                                               (isinstance(current_address, str) and len(new_address) > len(current_address))):
                                location_data['initial_address'] = location_data.get('address', 'Unknown')
                                location_data['address'] = new_address
                                location_data['address_source'] = 'initial_visual_analysis'
                                logger.info(f"Updated address from initial visual analysis: {new_address}")
                    
                    # Add visual clues to location data
                    if visual_analysis_results:
                        location_data['initial_visual_clues'] = visual_analysis_results.get('visual_analysis', {})
                        
                except TimeoutError as te:
                    logger.warning(f"Initial vision analysis timed out: {str(te)} - will complete in background")
                    visual_analysis_results = None
                    
            except Exception as vision_error:
                logger.error(f"Error in initial visual analysis: {str(vision_error)}")
                visual_analysis_results = None
                
        else:
            logger.info("Vision analysis will be completed in background processing")
        
        # Generate AI-powered neighborhood insights if available
        # NOTE: Temporarily disabled to prevent timeout issues - will be re-enabled with async processing
        neighborhood_insights = None
        if False:  # Disabled temporarily: ai_neighborhood_insights and location_data.get('address') and location_data.get('latitude') and location_data.get('longitude'):
            try:
                logger.info(f"Generating AI neighborhood insights for {location_data['address']}")
                neighborhood_insights = ai_neighborhood_insights(
                    location_data['address'],
                    location_data['latitude'],
                    location_data['longitude']
                )
                logger.info("Successfully generated neighborhood insights")
            except Exception as ai_error:
                logger.error(f"Error generating neighborhood insights: {str(ai_error)}")
        
        # Add neighborhood insights to the location data
        if neighborhood_insights:
            location_data['neighborhood_insights'] = neighborhood_insights
        
        # Get Street View metadata if coordinates are available
        street_view_metadata = None
        if location_data.get('latitude') and location_data.get('longitude'):
            try:
                import signal
                
                # Set up timeout handler for Street View metadata (3 seconds max)
                def sv_timeout_handler(signum, frame):
                    raise TimeoutError("Street View metadata fetch timed out after 3 seconds")
                
                # Set alarm for 3 seconds
                old_sv_handler = signal.signal(signal.SIGALRM, sv_timeout_handler)
                signal.alarm(3)
                
                try:
                    logger.info(f"Fetching Street View metadata for {location_data['latitude']}, {location_data['longitude']}")
                    street_view_metadata = get_street_view_metadata(
                        location_data['latitude'],
                        location_data['longitude']
                    )
                    
                    # Cancel the alarm if we finished in time
                    signal.alarm(0)
                    
                    # Add street view metadata to location data
                    location_data['street_view_metadata'] = street_view_metadata
                    
                    if street_view_metadata.get('available'):
                        logger.info(f"Street View available - Panorama ID: {street_view_metadata.get('panorama_id')}")
                    else:
                        logger.info(f"Street View not available - Status: {street_view_metadata.get('status')}")
                        
                except TimeoutError:
                    logger.warning("Street View metadata fetch timed out - proceeding without Street View data")
                    street_view_metadata = {
                        'available': False,
                        'status': 'TIMEOUT',
                        'error_message': 'Request timed out'
                    }
                    location_data['street_view_metadata'] = street_view_metadata
                finally:
                    # Reset the signal handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_sv_handler)
                    
            except Exception as sv_error:
                logger.error(f"Error getting Street View metadata: {str(sv_error)}")
                # Provide default metadata on error
                street_view_metadata = {
                    'available': False,
                    'status': 'ERROR',
                    'error_message': str(sv_error)
                }
                location_data['street_view_metadata'] = street_view_metadata
        else:
            logger.info("No coordinates available, skipping Street View metadata")
        
        # Cross-reference with real estate databases (only if enabled)
        real_estate_matches = {}
        if ENABLE_REAL_ESTATE_SEARCH and location_data.get('address'):
            try:
                logger.info(f"Cross-referencing property with real estate databases: {location_data['address']}")
                searcher = RealEstateSearcher()
                
                # Prepare coordinates if available
                coords = None
                if location_data.get('latitude') and location_data.get('longitude'):
                    coords = (location_data['latitude'], location_data['longitude'])
                
                # Extract visual features if available
                visual_features = None
                if visual_analysis_results and visual_analysis_results.get('visual_analysis'):
                    visual_features = visual_analysis_results['visual_analysis'].get('property_features', {})
                elif location_data.get('visual_clues', {}).get('property_features'):
                    visual_features = location_data['visual_clues']['property_features']
                
                # Enhance address with NLP data for better matching
                enhanced_address = location_data['address']
                if nlp_data and nlp_data.get('street_names'):
                    # Use the highest confidence street name to enhance search
                    best_street = nlp_data['street_names'][0] if nlp_data['street_names'] else None
                    if best_street and best_street['confidence'] > 80:
                        # If we have a high-confidence street name, prioritize it
                        logger.info(f"Using NLP-extracted street '{best_street['street_name']}' for enhanced search")
                        # Store for reference
                        location_data['nlp_best_street'] = best_street['street_name']
                
                # Cross-reference the property (with NLP data for fallback)
                real_estate_matches = searcher.cross_reference_property(
                    address=enhanced_address,
                    coordinates=coords,
                    visual_features=visual_features,
                    nlp_data=nlp_data
                )
                
                logger.info(f"Found {len(real_estate_matches.get('zillow_matches', []))} Zillow matches and {len(real_estate_matches.get('realtor_matches', []))} Realtor matches")
                
            except Exception as re_error:
                logger.error(f"Error in real estate search: {str(re_error)}")
                real_estate_matches = {
                    'error': 'Unable to search real estate databases at this time',
                    'zillow_matches': [],
                    'realtor_matches': []
                }
        elif not ENABLE_REAL_ESTATE_SEARCH:
            # Real estate search is disabled, indicate we're using advanced analysis
            logger.info("Real estate search disabled - Using Advanced OCR & NLP Analysis")
            real_estate_matches = {
                'disabled': True,
                'message': 'Using Advanced Vision & Text Analysis',
                'zillow_matches': [],
                'realtor_matches': []
            }
        
        # Skip multi-signal scoring here - will be done in background
        multi_signal_result = None
        logger.info("Multi-signal scoring will be performed in background processing")
        
        # All multi-signal scoring code has been moved to background processing
        # to avoid the 30-second timeout constraint
        
        # Disabled multi-signal scoring code removed (moved to background processing)
        # (All multi-signal scoring logic has been moved to process_in_background function)
        
        # Prepare initial data to store in the database (minimal processing)
        import json
        result_data = {
            'location_data': location_data,
            'original_url': url,
            'analysis_mode': analysis_mode,  # Store the analysis mode (quick/thorough)
            'google_maps_api_key': os.environ.get('GOOGLE_MAPS_API_KEY', ''),
            'search_results': search_results,
            'real_estate_matches': real_estate_matches,
            'visual_analysis_results': None,  # Will be populated in background
            'multi_signal_result': None,  # Will be populated in background
            'ai_enabled': bool(ai_neighborhood_insights),
            'real_estate_enabled': ENABLE_REAL_ESTATE_SEARCH
        }
        
        # Create a new session and save to database with pending status
        verification_result = VerificationResult.create_session(url, result_data)
        verification_result.processing_status = 'pending'
        verification_result.processing_progress = json.dumps({'stage': 'initial', 'message': 'Starting advanced analysis...'})
        db.session.add(verification_result)
        db.session.commit()
        
        logger.info(f"Created new verification session: {verification_result.session_id}")
        
        # Start background processing thread
        background_thread = threading.Thread(
            target=process_in_background,
            args=(verification_result.session_id,),
            daemon=True
        )
        background_thread.start()
        logger.info(f"Started background processing thread for session {verification_result.session_id}")
        
        # Track successful extraction and overall processing time
        extraction_duration = time.time() - extraction_start_time
        verification_time.labels(stage='total').observe(extraction_duration)
        verification_counter.labels(status='success', mode='thorough').inc()
        logger.info(f"Extraction completed in {extraction_duration:.2f} seconds")
        
        # Redirect to the results page which will show processing status
        return redirect(url_for("results", session_id=verification_result.session_id))
    
    except TimeoutError:
        logger.error("Request timeout while fetching Airbnb data")
        verification_counter.labels(status='timeout', mode='thorough').inc()
        flash("The request took too long to complete. Please try again later.", "danger")
        return redirect(url_for("index"))
    except ConnectionError:
        logger.error("Network connection error")
        verification_counter.labels(status='connection_error', mode='thorough').inc()
        flash("Could not connect to the Airbnb website. Please check your internet connection and try again.", "danger")
        return redirect(url_for("index"))
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        verification_counter.labels(status='value_error', mode='thorough').inc()
        flash(f"Invalid data received: {str(ve)}", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error extracting location data: {str(e)}\n{error_traceback}")
        # Provide user-friendly error messages based on error type
        if "404" in str(e):
            flash("The Airbnb listing could not be found. It may have been removed or the URL is incorrect.", "warning")
        elif "403" in str(e) or "forbidden" in str(e).lower():
            flash("Access to the Airbnb listing was denied. Please try again later.", "warning")
        elif "api" in str(e).lower() and "key" in str(e).lower():
            flash("There was an issue with the API configuration. Please contact support.", "danger")
        else:
            flash(f"An unexpected error occurred: {str(e)[:100]}...", "danger")
        return redirect(url_for("index"))

def generate_verification_urls(location_data):
    """Generate verification URLs for different real estate platforms"""
    import urllib.parse
    import re
    
    verification_urls = {
        'zillow': '',
        'realtor': '',
        'google_maps': '',
        'google_street_view': ''
    }
    
    # Get the best available address
    address = location_data.get('exact_address') or location_data.get('address')
    
    if address:
        # Zillow URL - uses URL encoding
        verification_urls['zillow'] = f"https://www.zillow.com/homes/{urllib.parse.quote(address)}_rb/"
        
        # Realtor.com URL - uses hyphenated format
        # Remove special characters and replace spaces with hyphens
        realtor_address = address.replace(',', '').replace('.', '').replace(' ', '-')
        # Remove multiple hyphens
        realtor_address = re.sub(r'-+', '-', realtor_address)
        verification_urls['realtor'] = f"https://www.realtor.com/realestateandhomes-search/{realtor_address}"
    
    # If we have coordinates, generate map URLs
    lat = location_data.get('latitude')
    lng = location_data.get('longitude')
    
    if lat and lng:
        # Google Maps URL
        verification_urls['google_maps'] = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
        
        # Google Street View URL
        verification_urls['google_street_view'] = f"https://www.google.com/maps/@{lat},{lng},19z/data=!3m1!1e3"
    
    return verification_urls

@app.route("/quick_results/<session_id>", methods=["GET"])
def quick_results(session_id):
    """Display quick verification results with queue options."""
    try:
        import json
        # Retrieve the verification result from the database
        verification_result = VerificationResult.get_by_session(session_id)
        
        if not verification_result:
            logger.warning(f"Session not found: {session_id}")
            flash("Results not found. The session may have expired or the link is invalid.", "warning")
            return redirect(url_for("index"))
        
        # Get the result data
        result_data = verification_result.data
        location_data = result_data.get('location_data', {})
        confidence = result_data.get('confidence', 0)
        status = result_data.get('status', 'inconclusive')
        
        # Generate verification URLs for external sites
        verification_urls = generate_verification_urls(location_data)
        
        # Determine if we should show the queue button
        show_queue_button = confidence < 70
        
        # Get Google Maps API key from environment
        google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        
        return render_template("quick_result.html",
                             original_url=result_data.get('original_url', ''),
                             location_data=location_data,
                             confidence=confidence,
                             confidence_factors=result_data.get('confidence_factors', []),
                             evidence_quality=result_data.get('evidence_quality', {}),
                             evidence_count=result_data.get('evidence_count', 0),
                             status=status,
                             show_queue_button=show_queue_button,
                             verification_urls=verification_urls,
                             processing_time=result_data.get('processing_time', 0),
                             session_id=session_id,
                             google_maps_api_key=google_maps_api_key)
        
    except Exception as e:
        logger.error(f"Error retrieving quick results: {str(e)}")
        flash("An error occurred while retrieving the results.", "danger")
        return redirect(url_for("index"))


@app.route("/quick_verify", methods=["GET", "POST"])
def quick_verify_handler():
    """Handle quick verification requests with enhanced error handling."""
    if request.method == "GET":
        return render_template("quick_verify.html")
    
    try:
        airbnb_url = request.form.get("airbnb_url", "").strip()
        
        # URL validation
        if not airbnb_url:
            flash("Please provide an Airbnb URL.", "warning")
            return redirect(url_for("quick_verify_handler"))
        
        # Validate URL format
        import re
        airbnb_pattern = re.compile(r'^https?://(www\.)?airbnb\.(com|[a-z]{2,3})/rooms/\d+', re.IGNORECASE)
        if not airbnb_pattern.match(airbnb_url):
            flash("Please enter a valid Airbnb listing URL (e.g., https://www.airbnb.com/rooms/123456)", "warning")
            return redirect(url_for("quick_verify_handler"))
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create verification result in database
        verification_result = VerificationResult(
            session_id=session_id,
            airbnb_url=airbnb_url,
            extracted_data=json.dumps({"status": "processing"}),
            processing_status="processing"
        )
        db.session.add(verification_result)
        db.session.commit()
        
        # Perform quick verification with timeout
        logger.info(f"Starting quick verification for {airbnb_url}")
        
        try:
            result = quick_verify_extraction(airbnb_url, timeout=10)
            
            # Check if listing was not found
            if result.get('status') == 'not_found':
                verification_result.processing_status = "not_found"
                verification_result.processing_error = result.get('message', 'Listing not found')
                verification_result.data = result
                db.session.commit()
                
                flash("This Airbnb listing is no longer available or has been removed.", "info")
                return redirect(url_for("index"))
            
            # Check for errors
            if not result.get('success'):
                error_msg = result.get('message', 'Verification failed')
                verification_result.processing_status = "error"
                verification_result.processing_error = error_msg
                verification_result.data = result
                db.session.commit()
                
                flash(f"Unable to verify this listing: {error_msg}", "warning")
                return redirect(url_for("quick_verify_handler"))
            
            # Add original URL to result
            result['original_url'] = airbnb_url
            
            # Store successful result
            verification_result.data = result
            verification_result.processing_status = "complete"
            verification_result.processing_error = None
            db.session.commit()
            
            # Log the verification
            confidence = result.get('confidence', 0)
            status = result.get('status', 'unknown')
            logger.info(f"Quick verification complete for {airbnb_url} - Status: {status}, Confidence: {confidence}%")
            
            # Add appropriate flash message based on confidence
            if confidence >= 75:
                flash(f"✓ Location verified with {confidence}% confidence!", "success")
            elif confidence >= 50:
                flash(f"Location identified with {confidence}% confidence. Consider deep analysis for better accuracy.", "info")
            else:
                flash(f"Low confidence result ({confidence}%). We recommend queuing for deep analysis.", "warning")
            
            # Redirect to results page
            return redirect(url_for("quick_results", session_id=session_id))
            
        except TimeoutError:
            verification_result.processing_status = "timeout"
            verification_result.processing_error = "Verification timed out. Please try again."
            db.session.commit()
            
            flash("The verification process took too long. Please try again or use Deep Analysis mode.", "warning")
            return redirect(url_for("quick_verify_handler"))
        
        except Exception as ve:
            # Specific verification error
            logger.error(f"Verification error for {airbnb_url}: {ve}")
            verification_result.processing_status = "error"
            verification_result.processing_error = str(ve)
            db.session.commit()
            
            flash(f"Verification error: {str(ve)}", "danger")
            return redirect(url_for("quick_verify_handler"))
        
    except Exception as e:
        logger.error(f"Error in quick verification handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        flash("An unexpected error occurred. Please try again or contact support.", "danger")
        return redirect(url_for("quick_verify_handler"))

@app.route("/queue_double_check", methods=["POST"])
def queue_double_check():
    """Add a double-check job to the processing queue."""
    try:
        airbnb_url = request.form.get('airbnb_url', '').strip()
        session_id = request.form.get('session_id', '').strip()
        
        if not airbnb_url:
            flash('URL is required', 'error')
            return redirect(url_for('index'))
        
        # Check if there's already a pending double-check job for this URL
        existing_job = JobQueue.query.filter_by(
            airbnb_url=airbnb_url,
            queue_type='double_check',
            status='pending'
        ).first()
        
        if existing_job:
            flash('This URL is already queued for double-checking', 'warning')
        else:
            # Create a new job with queue_type='double_check'
            # Generate a batch_id for grouping batch verification jobs
            batch_id = str(uuid.uuid4())
            job = JobQueue.create_job(
                airbnb_url=airbnb_url,
                priority=3,  # Higher priority than regular deep analysis
                analysis_mode='deep',  # Always use deep analysis for double-checks
                queue_type='double_check',
                batch_id=batch_id,
                queued_by=session_id or 'anonymous'
            )
            db.session.add(job)
            db.session.commit()
            flash(f'Successfully queued for double-checking', 'success')
            logger.info(f"Queued double-check job {job.job_id} for URL: {airbnb_url}")
        
        # Redirect back to the result page if we have a session_id
        if session_id:
            return redirect(url_for('view_result', session_id=session_id))
        else:
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error queuing double-check job: {e}")
        flash('Failed to queue for double-checking', 'error')
        return redirect(url_for('index'))

@app.route("/queue_for_deep_analysis", methods=["POST"])
def queue_for_deep_analysis():
    """Queue a quick verification result for comprehensive deep analysis."""
    try:
        airbnb_url = request.form.get('airbnb_url', '').strip()
        session_id = request.form.get('session_id', '').strip()
        
        if not airbnb_url:
            flash('URL is required', 'error')
            return redirect(url_for('index'))
        
        # Check if there's already a pending job for this URL
        existing_job = JobQueue.query.filter_by(
            airbnb_url=airbnb_url,
            status='pending'
        ).first()
        
        if existing_job:
            flash('This URL is already queued for analysis', 'warning')
        else:
            # Create a new deep analysis job
            job = JobQueue.create_job(
                airbnb_url=airbnb_url,
                priority=4,  # Standard priority for deep analysis
                analysis_mode='thorough',
                queue_type='deep_analysis',
                queued_by=session_id or 'quick_verify_upgrade'
            )
            db.session.add(job)
            db.session.commit()
            flash('Successfully queued for deep analysis. You will be notified when complete.', 'success')
            logger.info(f"Queued deep analysis job {job.job_id} for URL: {airbnb_url}")
        
        # Redirect back to quick results
        if session_id:
            return redirect(url_for('quick_results', session_id=session_id))
        else:
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error queuing deep analysis job: {e}")
        flash('Failed to queue for deep analysis', 'error')
        return redirect(url_for('index'))

@app.route("/queue_job", methods=["POST"])
def queue_job():
    """Queue a job for deep analysis."""
    try:
        url = request.form.get("airbnb_url", "").strip()
        notes = request.form.get("notes", "").strip()
        session_id = request.form.get("session_id", "").strip()
        
        if not url:
            return jsonify({"success": False, "message": "URL is required"}), 400
        
        # Check if job already exists in queue
        existing_job = JobQueue.query.filter_by(airbnb_url=url, status='pending').first()
        if existing_job:
            return jsonify({
                "success": True,
                "message": "This URL is already queued for analysis",
                "job_id": existing_job.job_id,
                "position": JobQueue.query.filter_by(status='pending').filter(
                    JobQueue.created_at <= existing_job.created_at).count()
            })
        
        # Create new job
        job = JobQueue.create_job(
            airbnb_url=url,
            priority=5,  # Default priority
            analysis_mode='deep',
            queued_by=session_id or 'anonymous',
            notes=notes
        )
        db.session.add(job)
        db.session.commit()
        
        # Get queue position
        position = JobQueue.query.filter_by(status='pending').filter(
            JobQueue.created_at <= job.created_at).count()
        
        logger.info(f"Queued job {job.job_id} for URL: {url}")
        
        return jsonify({
            "success": True,
            "message": "Job queued successfully for deep analysis",
            "job_id": job.job_id,
            "position": position
        })
        
    except Exception as e:
        logger.error(f"Error queuing job: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/queue_status", methods=["GET"])
def queue_status():
    """Get current queue statistics."""
    try:
        stats = JobQueue.get_queue_stats()
        
        # Get recent jobs if requested
        include_recent = request.args.get('recent', 'false').lower() == 'true'
        recent_jobs = []
        
        if include_recent:
            jobs = JobQueue.query.order_by(JobQueue.created_at.desc()).limit(10).all()
            for job in jobs:
                recent_jobs.append({
                    'job_id': job.job_id,
                    'url': job.airbnb_url[:50] + '...' if len(job.airbnb_url) > 50 else job.airbnb_url,
                    'status': job.status,
                    'created_at': job.created_at.isoformat(),
                    'notes': job.notes[:50] if job.notes else None
                })
        
        return jsonify({
            "success": True,
            "stats": stats,
            "recent_jobs": recent_jobs
        })
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/worker_status", methods=["GET"])
def worker_status():
    """Check if background workers are running and healthy."""
    import json
    import glob
    from datetime import datetime
    
    try:
        # Look for worker status files in /tmp
        status_files = glob.glob("/tmp/worker_*.status")
        workers = []
        
        for status_file in status_files:
            try:
                with open(status_file, 'r') as f:
                    worker_data = json.load(f)
                    
                # Check if heartbeat is recent (within last 2 minutes)
                last_heartbeat = datetime.fromisoformat(worker_data['last_heartbeat'])
                time_since_heartbeat = (datetime.utcnow() - last_heartbeat).total_seconds()
                
                worker_data['healthy'] = time_since_heartbeat < 120  # 2 minutes
                worker_data['time_since_heartbeat'] = int(time_since_heartbeat)
                workers.append(worker_data)
                
            except Exception as e:
                logger.warning(f"Could not read worker status file {status_file}: {e}")
        
        # Get queue statistics as well
        stats = JobQueue.get_queue_stats()
        
        # Determine overall status
        any_healthy = any(w.get('healthy', False) for w in workers)
        overall_status = 'running' if any_healthy else ('degraded' if workers else 'stopped')
        
        return jsonify({
            "success": True,
            "status": overall_status,
            "workers": workers,
            "worker_count": len(workers),
            "healthy_workers": sum(1 for w in workers if w.get('healthy', False)),
            "queue_stats": stats,
            "message": f"{len(workers)} worker(s) found, {sum(1 for w in workers if w.get('healthy', False))} healthy"
        })
        
    except Exception as e:
        logger.error(f"Error checking worker status: {str(e)}")
        return jsonify({
            "success": False,
            "status": "error",
            "message": str(e),
            "workers": [],
            "worker_count": 0
        }), 500


@app.route("/results/<session_id>", methods=["GET"])
def results(session_id):
    """Display verification results from a stored session or processing status."""
    try:
        import json
        # Retrieve the verification result from the database
        verification_result = VerificationResult.get_by_session(session_id)
        
        if not verification_result:
            logger.warning(f"Session not found: {session_id}")
            flash("Results not found. The session may have expired or the link is invalid.", "warning")
            return redirect(url_for("index"))
        
        # Check processing status
        status = verification_result.processing_status
        
        # If still processing, show processing page
        if status in ['pending', 'processing']:
            progress_data = {}
            try:
                if verification_result.processing_progress:
                    progress_data = json.loads(verification_result.processing_progress)
            except:
                progress_data = {'stage': 'processing', 'message': 'Processing...'}
            
            # Parse the stored data to get basic info
            result_data = verification_result.data
            location_data = result_data.get('location_data', {})
            
            return render_template("processing.html",
                                  session_id=session_id,
                                  location_data=location_data,
                                  status=status,
                                  progress=progress_data)
        
        # If error occurred, show error message
        if status == 'error':
            flash(f"An error occurred during processing: {verification_result.processing_error}", "danger")
            return redirect(url_for("index"))
        
        # Parse the stored data for completed processing
        result_data = verification_result.data
        
        # Extract the components for rendering
        location_data = result_data.get('location_data', {})
        original_url = result_data.get('original_url', '')
        # Get Google Maps API key from environment, fallback to result_data
        google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", result_data.get('google_maps_api_key', ''))
        search_results = result_data.get('search_results', {})
        real_estate_matches = result_data.get('real_estate_matches', {})
        visual_analysis_results = result_data.get('visual_analysis_results', None)
        ai_enabled = result_data.get('ai_enabled', False)
        real_estate_enabled = result_data.get('real_estate_enabled', ENABLE_REAL_ESTATE_SEARCH)
        
        # Generate verification URLs
        verification_urls = generate_verification_urls(location_data)
        
        # Render the results template with the stored data
        return render_template("result.html",
                              location_data=location_data,
                              original_url=original_url,
                              verification_urls=verification_urls,
                              google_maps_api_key=google_maps_api_key,
                              search_results=search_results,
                              real_estate_matches=real_estate_matches,
                              visual_analysis_results=visual_analysis_results,
                              ai_enabled=ai_enabled,
                              real_estate_enabled=real_estate_enabled)
    
    except Exception as e:
        logger.error(f"Error retrieving session results: {str(e)}")
        flash("An error occurred while retrieving the results. Please try again.", "danger")
        return redirect(url_for("index"))

@app.route("/api/status/<session_id>", methods=["GET"])
def check_status(session_id):
    """API endpoint to check processing status."""
    try:
        import json
        verification_result = VerificationResult.get_by_session(session_id)
        
        if not verification_result:
            return {'status': 'not_found', 'error': 'Session not found'}, 404
        
        progress_data = {}
        if verification_result.processing_progress:
            try:
                progress_data = json.loads(verification_result.processing_progress)
            except:
                progress_data = {}
        
        return {
            'status': verification_result.processing_status,
            'progress': progress_data,
            'error': verification_result.processing_error
        }
    
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return {'status': 'error', 'error': str(e)}, 500

@app.route("/queue_dashboard", methods=["GET"])
def queue_dashboard():
    """Display the job queue dashboard with statistics and job list."""
    try:
        # Get filter parameters
        status_filter = request.args.get('status', 'all')
        priority_filter = request.args.get('priority', 'all')
        page = request.args.get('page', 1, type=int)
        per_page = 25  # Jobs per page
        
        # Get queue statistics
        stats = JobQueue.get_queue_stats()
        
        # Calculate performance metrics
        # Get completed jobs from last 24 hours for metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        completed_jobs = JobQueue.query.filter(
            JobQueue.status == 'completed',
            JobQueue.completed_at > cutoff_time
        ).all()
        
        # Calculate average processing time
        processing_times = []
        for job in completed_jobs:
            if job.started_at and job.completed_at:
                delta = (job.completed_at - job.started_at).total_seconds()
                processing_times.append(delta)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate success rate
        from sqlalchemy import or_
        total_finished = JobQueue.query.filter(
            JobQueue.completed_at > cutoff_time,
            or_(JobQueue.status == 'completed', JobQueue.status == 'failed')
        ).count()
        
        success_rate = (len(completed_jobs) / total_finished * 100) if total_finished > 0 else 0
        
        # Calculate throughput
        hours_elapsed = min(24, (datetime.utcnow() - cutoff_time).total_seconds() / 3600)
        jobs_per_hour = len(completed_jobs) / hours_elapsed if hours_elapsed > 0 else 0
        
        # Build query for job list
        query = JobQueue.query
        
        # Apply filters
        if status_filter != 'all':
            query = query.filter_by(status=status_filter)
        
        if priority_filter != 'all':
            query = query.filter_by(priority=int(priority_filter))
        
        # Order by priority and creation time
        query = query.order_by(JobQueue.priority.asc(), JobQueue.created_at.desc())
        
        # Paginate results
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        jobs = pagination.items
        
        # Format performance metrics for display
        performance_metrics = {
            'avg_processing_time': f"{int(avg_processing_time // 60)}m {int(avg_processing_time % 60)}s" if avg_processing_time else "N/A",
            'success_rate': f"{success_rate:.1f}%",
            'jobs_per_hour': f"{jobs_per_hour:.1f}"
        }
        
        # Get double-check jobs for Batch Verification tab
        double_check_jobs = JobQueue.query.filter_by(
            queue_type='double_check',
            status='pending'
        ).order_by(JobQueue.created_at.asc()).all()
        
        double_check_count = len(double_check_jobs)
        
        return render_template("queue_dashboard.html",
                              jobs=jobs,
                              stats=stats,
                              performance_metrics=performance_metrics,
                              status_filter=status_filter,
                              priority_filter=priority_filter,
                              pagination=pagination,
                              double_check_jobs=double_check_jobs,
                              double_check_count=double_check_count)
    
    except Exception as e:
        logger.error(f"Error loading queue dashboard: {str(e)}")
        flash("Error loading queue dashboard", "danger")
        return redirect(url_for("index"))


@app.route("/start_batch_verification", methods=["POST"])
def start_batch_verification():
    """Start batch verification for all pending double-check jobs."""
    try:
        # Get all pending double-check jobs
        double_check_jobs = JobQueue.query.filter_by(
            queue_type='double_check',
            status='pending'
        ).order_by(JobQueue.created_at.asc()).all()
        
        if not double_check_jobs:
            return jsonify({
                'success': False,
                'message': 'No pending double-check jobs found'
            })
        
        # Generate a new batch_id for this batch
        batch_id = str(uuid.uuid4())
        
        # Update all jobs with the same batch_id
        for job in double_check_jobs:
            job.batch_id = batch_id
            # Mark them as ready for batch processing
            job.priority = 2  # Higher priority for batch jobs
        
        db.session.commit()
        
        # Start the batch processing in a background thread
        batch_thread = threading.Thread(
            target=process_batch_verification,
            args=(batch_id, len(double_check_jobs)),
            daemon=True
        )
        batch_thread.start()
        
        logger.info(f"Started batch verification for {len(double_check_jobs)} jobs with batch_id: {batch_id}")
        
        return jsonify({
            'success': True,
            'job_count': len(double_check_jobs),
            'batch_id': batch_id,
            'message': f'Started batch verification for {len(double_check_jobs)} jobs'
        })
        
    except Exception as e:
        logger.error(f"Error starting batch verification: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })


def process_batch_verification(batch_id, job_count):
    """Process batch verification jobs with delays between each job."""
    import random
    
    with app.app_context():
        logger.info(f"Starting batch processing for batch_id: {batch_id}")
        
        # Get all jobs in this batch
        batch_jobs = JobQueue.query.filter_by(
            batch_id=batch_id,
            queue_type='double_check'
        ).order_by(JobQueue.created_at.asc()).all()
        
        for index, job in enumerate(batch_jobs, 1):
            try:
                # Skip if job is no longer pending
                if job.status != 'pending':
                    logger.info(f"Skipping job {job.job_id} - status is {job.status}")
                    continue
                
                logger.info(f"Processing batch job {index}/{job_count}: {job.job_id}")
                
                # Mark job as ready for immediate processing (highest priority)
                job.priority = 1  # Highest priority
                job.last_activity = datetime.utcnow()
                db.session.commit()
                
                # If this is not the last job, add a delay
                if index < job_count:
                    # Random delay between 5-10 minutes (300-600 seconds)
                    delay = random.randint(300, 600)
                    logger.info(f"Waiting {delay} seconds before next job...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch job {job.job_id}: {e}")
                continue
        
        logger.info(f"Batch processing completed for batch_id: {batch_id}")


@app.route("/queue_job/<job_id>", methods=["GET"])
def job_details(job_id):
    """Display detailed information about a specific job."""
    try:
        # Get the job
        job = JobQueue.query.filter_by(job_id=job_id).first()
        
        if not job:
            flash("Job not found", "warning")
            return redirect(url_for("queue_dashboard"))
        
        # Get verification result if completed
        verification_result = None
        if job.verification_result_id:
            verification_result = VerificationResult.query.get(job.verification_result_id)
        
        # Calculate processing time
        processing_time = None
        if job.started_at and job.completed_at:
            delta = (job.completed_at - job.started_at).total_seconds()
            processing_time = f"{int(delta // 60)}m {int(delta % 60)}s"
        elif job.started_at:
            delta = (datetime.utcnow() - job.started_at).total_seconds()
            processing_time = f"{int(delta // 60)}m {int(delta % 60)}s (ongoing)"
        
        return render_template("job_details.html",
                              job=job,
                              verification_result=verification_result,
                              processing_time=processing_time)
    
    except Exception as e:
        logger.error(f"Error loading job details: {str(e)}")
        flash("Error loading job details", "danger")
        return redirect(url_for("queue_dashboard"))


@app.route("/queue_job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a pending job."""
    try:
        job = JobQueue.query.filter_by(job_id=job_id).first()
        
        if not job:
            flash("Job not found", "warning")
        elif job.status != 'pending':
            flash("Only pending jobs can be cancelled", "warning")
        else:
            job.status = 'cancelled'
            job.completed_at = datetime.utcnow()
            db.session.commit()
            flash("Job cancelled successfully", "success")
        
        return redirect(url_for("job_details", job_id=job_id))
    
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        flash("Error cancelling job", "danger")
        return redirect(url_for("job_details", job_id=job_id))


@app.route("/queue_job/<job_id>/retry", methods=["POST"])
def retry_job(job_id):
    """Retry a failed job."""
    try:
        job = JobQueue.query.filter_by(job_id=job_id).first()
        
        if not job:
            flash("Job not found", "warning")
        elif job.status not in ['failed', 'cancelled']:
            flash("Only failed or cancelled jobs can be retried", "warning")
        else:
            job.status = 'pending'
            job.retry_count += 1
            job.started_at = None
            job.completed_at = None
            job.worker_id = None
            job.error_message = None
            job.last_activity = datetime.utcnow()
            db.session.commit()
            flash("Job queued for retry", "success")
        
        return redirect(url_for("job_details", job_id=job_id))
    
    except Exception as e:
        logger.error(f"Error retrying job: {str(e)}")
        flash("Error retrying job", "danger")
        return redirect(url_for("job_details", job_id=job_id))


@app.route("/queue_job/<job_id>/priority", methods=["POST"])
def change_priority(job_id):
    """Change the priority of a job."""
    try:
        new_priority = request.form.get('priority', type=int)
        
        if not new_priority or new_priority < 1 or new_priority > 10:
            flash("Invalid priority. Please enter a value between 1 and 10", "warning")
            return redirect(url_for("job_details", job_id=job_id))
        
        job = JobQueue.query.filter_by(job_id=job_id).first()
        
        if not job:
            flash("Job not found", "warning")
        elif job.status not in ['pending']:
            flash("Only pending jobs can have their priority changed", "warning")
        else:
            job.priority = new_priority
            job.last_activity = datetime.utcnow()
            db.session.commit()
            flash(f"Priority changed to {new_priority}", "success")
        
        return redirect(url_for("job_details", job_id=job_id))
    
    except Exception as e:
        logger.error(f"Error changing priority: {str(e)}")
        flash("Error changing job priority", "danger")
        return redirect(url_for("job_details", job_id=job_id))


@app.route("/queue/bulk_action", methods=["POST"])
def bulk_action():
    """Perform bulk actions on selected jobs."""
    try:
        action = request.form.get('action')
        job_ids = request.form.getlist('job_ids[]')
        
        if not job_ids:
            flash("No jobs selected", "warning")
            return redirect(url_for("queue_dashboard"))
        
        # Get all selected jobs
        jobs = JobQueue.query.filter(JobQueue.job_id.in_(job_ids)).all()
        
        if action == 'cancel':
            count = 0
            for job in jobs:
                if job.status == 'pending':
                    job.status = 'cancelled'
                    job.completed_at = datetime.utcnow()
                    count += 1
            db.session.commit()
            flash(f"Cancelled {count} pending jobs", "success")
        
        elif action == 'retry':
            count = 0
            for job in jobs:
                if job.status in ['failed', 'cancelled']:
                    job.status = 'pending'
                    job.retry_count += 1
                    job.started_at = None
                    job.completed_at = None
                    job.worker_id = None
                    job.error_message = None
                    job.last_activity = datetime.utcnow()
                    count += 1
            db.session.commit()
            flash(f"Queued {count} jobs for retry", "success")
        
        elif action == 'delete':
            count = 0
            for job in jobs:
                if job.status in ['completed', 'failed', 'cancelled']:
                    db.session.delete(job)
                    count += 1
            db.session.commit()
            flash(f"Deleted {count} jobs", "success")
        
        else:
            flash("Invalid action", "warning")
        
        return redirect(url_for("queue_dashboard"))
    
    except Exception as e:
        logger.error(f"Error performing bulk action: {str(e)}")
        flash("Error performing bulk action", "danger")
        return redirect(url_for("queue_dashboard"))


@app.route("/queue/export", methods=["GET"])
def export_queue():
    """Export queue data to CSV."""
    try:
        # Get all jobs or filter based on parameters
        status_filter = request.args.get('status', 'all')
        
        query = JobQueue.query
        if status_filter != 'all':
            query = query.filter_by(status=status_filter)
        
        jobs = query.order_by(JobQueue.created_at.desc()).all()
        
        # Create CSV
        si = StringIO()
        writer = csv.writer(si)
        
        # Write headers
        writer.writerow([
            'Job ID', 'URL', 'Status', 'Priority', 'Analysis Mode',
            'Created', 'Started', 'Completed', 'Worker ID',
            'Error Message', 'Retry Count', 'Notes'
        ])
        
        # Write job data
        for job in jobs:
            writer.writerow([
                job.job_id,
                job.airbnb_url,
                job.status,
                job.priority,
                job.analysis_mode,
                job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else '',
                job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else '',
                job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else '',
                job.worker_id or '',
                job.error_message or '',
                job.retry_count,
                job.notes or ''
            ])
        
        # Create response
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=queue_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        output.headers["Content-type"] = "text/csv"
        
        return output
    
    except Exception as e:
        logger.error(f"Error exporting queue data: {str(e)}")
        flash("Error exporting queue data", "danger")
        return redirect(url_for("queue_dashboard"))


@app.route("/queue/clear_old", methods=["POST"])
def clear_old_jobs():
    """Clear completed jobs older than specified days."""
    try:
        days = request.form.get('days', 7, type=int)
        
        if days < 1:
            flash("Invalid number of days", "warning")
            return redirect(url_for("queue_dashboard"))
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Delete old completed jobs
        from sqlalchemy import or_
        deleted = JobQueue.query.filter(
            or_(JobQueue.status == 'completed', JobQueue.status == 'cancelled'),
            JobQueue.completed_at < cutoff_time
        ).delete()
        
        db.session.commit()
        
        flash(f"Deleted {deleted} jobs older than {days} days", "success")
        return redirect(url_for("queue_dashboard"))
    
    except Exception as e:
        logger.error(f"Error clearing old jobs: {str(e)}")
        flash("Error clearing old jobs", "danger")
        return redirect(url_for("queue_dashboard"))


@app.route("/api/queue/stats", methods=["GET"])
def api_queue_stats():
    """API endpoint to get queue statistics for auto-refresh."""
    try:
        stats = JobQueue.get_queue_stats()
        
        # Get performance metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        completed_jobs = JobQueue.query.filter(
            JobQueue.status == 'completed',
            JobQueue.completed_at > cutoff_time
        ).all()
        
        # Calculate average processing time
        processing_times = []
        for job in completed_jobs:
            if job.started_at and job.completed_at:
                delta = (job.completed_at - job.started_at).total_seconds()
                processing_times.append(delta)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate success rate
        from sqlalchemy import or_
        total_finished = JobQueue.query.filter(
            JobQueue.completed_at > cutoff_time,
            or_(JobQueue.status == 'completed', JobQueue.status == 'failed')
        ).count()
        
        success_rate = (len(completed_jobs) / total_finished * 100) if total_finished > 0 else 0
        
        # Calculate throughput
        hours_elapsed = min(24, (datetime.utcnow() - cutoff_time).total_seconds() / 3600)
        jobs_per_hour = len(completed_jobs) / hours_elapsed if hours_elapsed > 0 else 0
        
        return jsonify({
            'stats': stats,
            'metrics': {
                'avg_processing_time': int(avg_processing_time),
                'success_rate': round(success_rate, 1),
                'jobs_per_hour': round(jobs_per_hour, 1)
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting queue stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/monitoring", methods=["GET"])
def monitoring_dashboard():
    """Display real-time monitoring dashboard with Prometheus metrics."""
    try:
        # Get metrics summary
        from core.metrics import get_metrics_summary, alert_manager
        metrics_summary = get_metrics_summary()
        
        # Get recent alerts
        alerts = []
        
        # Check for high error rate
        if metrics_summary.get('external_apis', {}).get('total_errors', 0) > 100:
            alerts.append({
                'type': 'warning',
                'message': 'High API error rate detected',
                'metric': 'external_apis.total_errors',
                'value': metrics_summary['external_apis']['total_errors']
            })
        
        # Check for high CPU usage
        cpu_usage = metrics_summary.get('system', {}).get('cpu_usage', 0)
        if cpu_usage > 80:
            alerts.append({
                'type': 'warning',
                'message': 'High CPU usage detected',
                'metric': 'system.cpu_usage',
                'value': f"{cpu_usage:.1f}%"
            })
        
        # Check for high memory usage
        memory_usage = metrics_summary.get('system', {}).get('memory_usage', 0)
        if memory_usage > 90:
            alerts.append({
                'type': 'danger',
                'message': 'Critical memory usage',
                'metric': 'system.memory_usage',
                'value': f"{memory_usage:.1f}%"
            })
        
        # Check for job queue backlog
        active_jobs = metrics_summary.get('background_jobs', {}).get('active', 0)
        if active_jobs > 50:
            alerts.append({
                'type': 'warning',
                'message': 'High number of active background jobs',
                'metric': 'background_jobs.active',
                'value': active_jobs
            })
        
        # Get recent verification stats from database
        from datetime import datetime, timedelta
        from core.models import VerificationResult, JobQueue
        
        # Stats for last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_verifications = VerificationResult.query.filter(
            VerificationResult.created_at > one_hour_ago
        ).count()
        
        # Success rate calculation
        completed_verifications = VerificationResult.query.filter(
            VerificationResult.created_at > one_hour_ago,
            VerificationResult.processing_status == 'completed'
        ).count()
        
        success_rate = (completed_verifications / recent_verifications * 100) if recent_verifications > 0 else 0
        
        # Job queue stats
        pending_jobs = JobQueue.query.filter_by(status='pending').count()
        processing_jobs = JobQueue.query.filter_by(status='processing').count()
        failed_jobs = JobQueue.query.filter(
            JobQueue.status == 'failed',
            JobQueue.completed_at > one_hour_ago
        ).count()
        
        # Prepare data for the template
        dashboard_data = {
            'metrics': metrics_summary,
            'alerts': alerts,
            'stats': {
                'recent_verifications': recent_verifications,
                'success_rate': round(success_rate, 1),
                'pending_jobs': pending_jobs,
                'processing_jobs': processing_jobs,
                'failed_jobs': failed_jobs
            }
        }
        
        return render_template("monitoring_dashboard.html", **dashboard_data)
    
    except Exception as e:
        logger.error(f"Error loading monitoring dashboard: {str(e)}")
        flash("Error loading monitoring dashboard", "danger")
        return redirect(url_for("index"))


@app.route("/api/monitoring/metrics", methods=["GET"])
def monitoring_metrics_api():
    """API endpoint for real-time metrics updates."""
    try:
        from core.metrics import get_metrics_summary
        metrics = get_metrics_summary()
        
        # Add timestamp
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template("index.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    import traceback
    error_traceback = traceback.format_exc()
    logger.error(f"Server error (500): {str(e)}\nFull traceback:\n{error_traceback}")
    
    # In debug mode, show the actual error
    if app.debug:
        return f"<h1>Internal Server Error</h1><pre>{error_traceback}</pre>", 500
    
    flash("An internal server error occurred. Please try again later.", "danger")
    return render_template("index.html", error="Server error"), 500

# Main entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)