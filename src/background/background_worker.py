#!/usr/bin/env python3
"""
Background Worker for Processing Airbnb Verification Jobs

This worker continuously processes queued verification jobs with proper
rate limiting, error handling, and anti-detection mechanisms.
"""

import os
import sys
import logging
import signal
import time
import random
import json
import uuid
import psutil
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Add parent directory to path for imports (src directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Flask app and models
from core.app import app, db, vision_analyzer
from core.models import JobQueue, VerificationResult
from extraction.scraper import get_airbnb_location_data
from scoring.multi_signal_scorer import select_best_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('worker.log')
    ]
)
logger = logging.getLogger(__name__)

# Worker configuration
WORKER_ID = f"worker_{uuid.uuid4().hex[:8]}"
MIN_DELAY = 30  # Minimum seconds between jobs
MAX_DELAY = 120  # Maximum seconds between jobs
STALE_JOB_TIMEOUT = 30  # Minutes before a job is considered stale
ERROR_BACKOFF_BASE = 60  # Base seconds for exponential backoff
MAX_ERROR_BACKOFF = 600  # Maximum backoff time (10 minutes)
HEARTBEAT_INTERVAL = 60  # Seconds between heartbeat updates

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True

def get_random_delay(min_seconds: int = MIN_DELAY, max_seconds: int = MAX_DELAY) -> float:
    """
    Get a random delay with jitter to avoid patterns.
    
    Args:
        min_seconds: Minimum delay in seconds
        max_seconds: Maximum delay in seconds
        
    Returns:
        Random delay time in seconds with additional jitter
    """
    base_delay = random.uniform(min_seconds, max_seconds)
    jitter = random.uniform(-5, 5)  # Add/subtract up to 5 seconds
    return max(min_seconds, base_delay + jitter)

def is_worker_healthy() -> bool:
    """Check if worker is healthy and can continue processing."""
    try:
        # Check memory usage
        process = psutil.Process()
        memory_percent = process.memory_percent()
        if memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
            return False
        
        # Check if database is accessible
        with app.app_context():
            db.session.execute(db.text("SELECT 1"))
            db.session.commit()
        
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def process_job_thorough(job: JobQueue) -> Dict[str, Any]:
    """
    Process a job in thorough/deep analysis mode.
    Target processing time: 30-45 seconds with enforceable deadlines.
    
    Args:
        job: The JobQueue object to process
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    import time
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    from scoring.streetview_matcher import StreetViewMatcher
    from scoring.real_estate_searcher import RealEstateSearcher
    from extraction.scraper import get_street_view_metadata
    
    start_time = time.time()
    logger.info(f"[{WORKER_ID}] Starting DEEP ANALYSIS for job {job.job_id}")
    logger.info(f"[{WORKER_ID}] Target processing time: 30-45 seconds with strict deadlines")
    
    # Define timing budgets for each stage (in seconds)
    TOTAL_DEADLINE = 45  # Hard deadline
    STAGE_BUDGETS = {
        'scraping': 10,           # Stage 1: Initial scraping
        'vision_analysis': 15,     # Stage 2: AI vision analysis
        'street_view': 8,         # Stage 3: Street view processing
        'real_estate': 7,         # Stage 4: Real estate search
        'scoring': 5              # Stage 5: Multi-signal scoring
    }
    
    # Initialize tracking for evidence breakdown
    evidence_breakdown = {
        'signals_used': [],
        'processing_stages': [],
        'timing': {},
        'confidence_components': {},
        'deadline_enforced': False,
        'stages_skipped': []
    }
    
    # Helper function to check remaining time
    def get_remaining_time():
        elapsed = time.time() - start_time
        return max(0, TOTAL_DEADLINE - elapsed)
    
    # Helper function to check if we should skip a stage
    def should_skip_stage(stage_name: str):
        remaining = get_remaining_time()
        if remaining < 5:  # Less than 5 seconds left - skip non-essential stages
            logger.warning(f"[{WORKER_ID}] Skipping {stage_name} - approaching deadline (remaining: {remaining:.1f}s)")
            evidence_breakdown['stages_skipped'].append(stage_name)
            return True
        return False
    
    # Stage 1: Get full location data with all extraction methods (with timeout)
    stage_start = time.time()
    location_data = None
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_airbnb_location_data, job.airbnb_url)
            timeout = min(STAGE_BUDGETS['scraping'], get_remaining_time())
            location_data = future.result(timeout=timeout)
            evidence_breakdown['timing']['scraping'] = time.time() - stage_start
            evidence_breakdown['processing_stages'].append('scraping')
            logger.info(f"[{WORKER_ID}] Stage 1 - Scraping completed in {evidence_breakdown['timing']['scraping']:.2f}s")
    except (FutureTimeoutError, Exception) as e:
        logger.error(f"[{WORKER_ID}] Stage 1 - Scraping failed or timed out: {str(e)}")
        evidence_breakdown['timing']['scraping'] = time.time() - stage_start
        # Return early if scraping fails
        location_data = {'error': f"Scraping failed: {str(e)}", 'address': None}
        location_data['evidence_breakdown'] = evidence_breakdown
        return location_data
    
    # Initialize address candidates list
    all_address_candidates = []
    if location_data.get('address'):
        all_address_candidates.append({
            'address': location_data['address'],
            'source': 'airbnb_listing',
            'confidence': 60
        })
    
    # Stage 2: AI Vision Analysis (up to 15 photos) - with timing control
    visual_analysis_results = None
    if vision_analyzer and location_data.get('photos') and not should_skip_stage('vision_analysis'):
        # Check if Google Vision API is available
        google_vision_available = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None
        if not google_vision_available:
            logger.info(f"[{WORKER_ID}] Stage 2 - Vision analysis skipped (Google Vision API not configured)")
            evidence_breakdown['stages_skipped'].append('vision_analysis_no_api')
        else:
            stage_start = time.time()
            try:
                photos_to_analyze = location_data['photos'][:15]  # Analyze up to 15 photos for deep analysis
                logger.info(f"[{WORKER_ID}] Stage 2 - Analyzing {len(photos_to_analyze)} photos with AI vision")
                
                coords = None
                if location_data.get('latitude') and location_data.get('longitude'):
                    coords = (location_data['latitude'], location_data['longitude'])
                
                # Run vision analysis with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(vision_analyzer, photos_to_analyze, coords)
                    timeout = min(STAGE_BUDGETS['vision_analysis'], get_remaining_time())
                    visual_analysis_results = future.result(timeout=timeout)
                
                evidence_breakdown['timing']['vision_analysis'] = time.time() - stage_start
                evidence_breakdown['processing_stages'].append('vision_analysis')
                evidence_breakdown['signals_used'].append('ai_vision')
                
                # Extract visual clues and landmarks
                if visual_analysis_results:
                    location_data['visual_clues'] = visual_analysis_results.get('visual_analysis', {})
                    location_data['ocr_results'] = visual_analysis_results.get('ocr_summary', {})
                    location_data['landmarks'] = visual_analysis_results.get('landmarks', [])
                    location_data['building_features'] = visual_analysis_results.get('building_features', [])
                    
                    # Add visual address candidate if found
                    if visual_analysis_results.get('suggested_address'):
                        confidence = visual_analysis_results.get('final_address_confidence', 0)
                        all_address_candidates.append({
                            'address': visual_analysis_results['suggested_address'],
                            'source': 'ai_vision_ocr',
                            'confidence': confidence,
                            'ocr_evidence': visual_analysis_results.get('ocr_address_data', {})
                        })
                        evidence_breakdown['confidence_components']['ocr'] = confidence
                        
                logger.info(f"[{WORKER_ID}] Stage 2 - Vision analysis completed in {evidence_breakdown['timing']['vision_analysis']:.2f}s")
                    
            except FutureTimeoutError:
                logger.warning(f"[{WORKER_ID}] Stage 2 - Vision analysis timed out after {timeout:.1f}s")
                evidence_breakdown['timing']['vision_analysis'] = time.time() - stage_start
                evidence_breakdown['stages_skipped'].append('vision_analysis_timeout')
            except Exception as e:
                logger.error(f"[{WORKER_ID}] Error in visual analysis: {str(e)}")
                evidence_breakdown['timing']['vision_analysis'] = time.time() - stage_start
    
    # Stage 3: Street View Metadata and Matching - with timing control
    street_view_data = None
    if location_data.get('latitude') and location_data.get('longitude') and not should_skip_stage('street_view'):
        # Check if Street View API is available
        street_view_api_key = os.environ.get('GOOGLE_STREETVIEW_API_KEY') or os.environ.get('GOOGLE_MAPS_API_KEY')
        if not street_view_api_key:
            logger.info(f"[{WORKER_ID}] Stage 3 - Street View skipped (API key not configured)")
            evidence_breakdown['stages_skipped'].append('street_view_no_api')
        else:
            stage_start = time.time()
            try:
                logger.info(f"[{WORKER_ID}] Stage 3 - Fetching Street View metadata")
                
                # Get Street View metadata with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        get_street_view_metadata,
                        location_data['latitude'],
                        location_data['longitude']
                    )
                    timeout = min(STAGE_BUDGETS['street_view'], get_remaining_time())
                    street_view_data = future.result(timeout=timeout)
                
                location_data['street_view_metadata'] = street_view_data
                
                # Try Street View matching if available
                if street_view_data and street_view_data.get('available'):
                    evidence_breakdown['signals_used'].append('street_view')
                    logger.info(f"[{WORKER_ID}] Street View available - Panorama ID: {street_view_data.get('panorama_id')}")
                    
                    # Initialize Street View matcher for visual comparison
                    try:
                        # Check remaining time before attempting match
                        if get_remaining_time() > 3:  # Need at least 3 seconds for matching
                            matcher = StreetViewMatcher()
                            if location_data.get('photos') and len(location_data['photos']) > 0:
                                # Run matching with timeout
                                with ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(
                                        matcher.compare_with_streetview,
                                        listing_photos=location_data['photos'][:5],
                                        lat=location_data['latitude'],
                                        lng=location_data['longitude']
                                    )
                                    match_timeout = min(3, get_remaining_time())
                                    match_result = future.result(timeout=match_timeout)
                                    
                                    if match_result:
                                        location_data['street_view_match'] = match_result
                                        evidence_breakdown['confidence_components']['street_view'] = match_result.get('confidence', 0)
                        else:
                            logger.info(f"[{WORKER_ID}] Skipping Street View matching - insufficient time")
                            evidence_breakdown['stages_skipped'].append('street_view_matching')
                    except FutureTimeoutError:
                        logger.warning(f"[{WORKER_ID}] Street View matching timed out")
                        evidence_breakdown['stages_skipped'].append('street_view_match_timeout')
                    except Exception as sv_match_error:
                        logger.warning(f"[{WORKER_ID}] Street View matching error: {sv_match_error}")
                
                evidence_breakdown['timing']['street_view'] = time.time() - stage_start
                evidence_breakdown['processing_stages'].append('street_view')
                logger.info(f"[{WORKER_ID}] Stage 3 - Street View processing completed in {evidence_breakdown['timing']['street_view']:.2f}s")
                
            except FutureTimeoutError:
                logger.warning(f"[{WORKER_ID}] Stage 3 - Street View timed out after {timeout:.1f}s")
                evidence_breakdown['timing']['street_view'] = time.time() - stage_start
                evidence_breakdown['stages_skipped'].append('street_view_timeout')
            except Exception as e:
                logger.error(f"[{WORKER_ID}] Error in Street View processing: {str(e)}")
                evidence_breakdown['timing']['street_view'] = time.time() - stage_start
    
    # Stage 4: Optional Real Estate Cross-Reference - with timing control
    real_estate_data = None
    real_estate_enabled = os.environ.get("ENABLE_REAL_ESTATE_SEARCH", "false").lower() in ["true", "1", "yes", "on"]
    if real_estate_enabled and not should_skip_stage('real_estate'):
        stage_start = time.time()
        try:
            logger.info(f"[{WORKER_ID}] Stage 4 - Real estate cross-reference enabled")
            
            if location_data.get('address'):
                # Run real estate search with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    searcher = RealEstateSearcher()
                    future = executor.submit(searcher.search_property, location_data['address'])
                    timeout = min(STAGE_BUDGETS['real_estate'], get_remaining_time())
                    results = future.result(timeout=timeout)
                    
                    if results and results.get('properties'):
                        real_estate_data = results
                        location_data['real_estate_matches'] = results['properties'][:3]  # Top 3 matches
                        evidence_breakdown['signals_used'].append('real_estate')
                        
                        # Add real estate addresses as candidates
                        for prop in results['properties'][:3]:
                            if prop.get('address'):
                                all_address_candidates.append({
                                    'address': prop['address'],
                                    'source': 'real_estate_listing',
                                    'confidence': 70,
                                    'property_details': {
                                        'bedrooms': prop.get('bedrooms'),
                                        'bathrooms': prop.get('bathrooms'),
                                        'property_type': prop.get('property_type')
                                    }
                                })
                
                evidence_breakdown['timing']['real_estate'] = time.time() - stage_start
                evidence_breakdown['processing_stages'].append('real_estate')
                logger.info(f"[{WORKER_ID}] Stage 4 - Real estate search completed in {evidence_breakdown['timing']['real_estate']:.2f}s")
            else:
                logger.info(f"[{WORKER_ID}] Stage 4 - Real estate search skipped (no address available)")
                evidence_breakdown['stages_skipped'].append('real_estate_no_address')
                
        except FutureTimeoutError:
            logger.warning(f"[{WORKER_ID}] Stage 4 - Real estate search timed out after {timeout:.1f}s")
            evidence_breakdown['timing']['real_estate'] = time.time() - stage_start
            evidence_breakdown['stages_skipped'].append('real_estate_timeout')
        except Exception as e:
            logger.warning(f"[{WORKER_ID}] Real estate search error: {str(e)}")
            evidence_breakdown['timing']['real_estate'] = time.time() - stage_start
    elif not real_estate_enabled:
        logger.info(f"[{WORKER_ID}] Stage 4 - Real estate search disabled by configuration")
        evidence_breakdown['stages_skipped'].append('real_estate_disabled')
    
    # Stage 5: Multi-Signal Scoring and Address Ranking - with timing control
    stage_start = time.time()
    
    # Check if we should skip scoring due to deadline
    if should_skip_stage('scoring'):
        # Provide a basic result if we're out of time
        logger.warning(f"[{WORKER_ID}] Stage 5 - Scoring skipped due to deadline")
        if all_address_candidates:
            # Use the first candidate with highest confidence as fallback
            best_candidate = max(all_address_candidates, key=lambda x: x.get('confidence', 0))
            location_data['address'] = best_candidate['address']
            location_data['address_source'] = best_candidate.get('source', 'fallback')
            location_data['confidence_score'] = best_candidate.get('confidence', 50)
        evidence_breakdown['stages_skipped'].append('scoring_deadline')
    else:
        logger.info(f"[{WORKER_ID}] Stage 5 - Multi-signal scoring for {len(all_address_candidates)} candidates")
        
        # Store all candidates for ranking
        location_data['address_candidates'] = all_address_candidates
        
        # Perform comprehensive multi-signal scoring with timeout
        if all_address_candidates:
            try:
                # Check if APIs are available to adjust scoring weights
                google_vision_available = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None
                real_estate_enabled_for_scoring = real_estate_enabled
                
                # Run scoring with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        select_best_address,
                        all_address_candidates,
                        location_data.get('latitude'),
                        location_data.get('longitude'),
                        ocr_data=location_data.get('ocr_results'),
                        nlp_data=location_data.get('nlp_extraction'),
                        vision_features=location_data.get('building_features'),
                        google_vision_available=google_vision_available,
                        real_estate_enabled=real_estate_enabled_for_scoring
                    )
                    timeout = min(STAGE_BUDGETS['scoring'], get_remaining_time())
                    best_address = future.result(timeout=timeout)
                
                if best_address:
                    location_data['best_address'] = best_address
                    location_data['address'] = best_address['address']
                    location_data['confidence_score'] = best_address.get('final_score', 0)
                    location_data['address_source'] = best_address.get('source', 'multi_signal')
                    
                    # Add confidence breakdown
                    if best_address.get('multi_signal_scoring'):
                        evidence_breakdown['confidence_components'] = best_address['multi_signal_scoring'].get('contributions', {})
                
                evidence_breakdown['timing']['scoring'] = time.time() - stage_start
                evidence_breakdown['processing_stages'].append('multi_signal_scoring')
                evidence_breakdown['signals_used'].append('multi_signal_scoring')
                
            except FutureTimeoutError:
                logger.warning(f"[{WORKER_ID}] Stage 5 - Scoring timed out after {timeout:.1f}s")
                # Use fallback scoring
                if all_address_candidates:
                    best_candidate = max(all_address_candidates, key=lambda x: x.get('confidence', 0))
                    location_data['address'] = best_candidate['address']
                    location_data['address_source'] = best_candidate.get('source', 'timeout_fallback')
                    location_data['confidence_score'] = best_candidate.get('confidence', 50)
                evidence_breakdown['timing']['scoring'] = time.time() - stage_start
                evidence_breakdown['stages_skipped'].append('scoring_timeout')
            except Exception as e:
                logger.error(f"[{WORKER_ID}] Error in multi-signal scoring: {str(e)}")
                evidence_breakdown['timing']['scoring'] = time.time() - stage_start
    
    # Calculate total processing time
    total_time = time.time() - start_time
    evidence_breakdown['timing']['total'] = total_time
    
    # Add comprehensive evidence breakdown to results
    location_data['evidence_breakdown'] = evidence_breakdown
    location_data['processing_mode'] = 'deep'
    location_data['worker_id'] = WORKER_ID
    location_data['processed_at'] = datetime.utcnow().isoformat()
    location_data['processing_time_seconds'] = total_time
    
    # Log summary
    logger.info(f"[{WORKER_ID}] DEEP ANALYSIS COMPLETE for job {job.job_id}")
    logger.info(f"[{WORKER_ID}] Total processing time: {total_time:.2f}s")
    logger.info(f"[{WORKER_ID}] Signals used: {', '.join(evidence_breakdown['signals_used'])}")
    logger.info(f"[{WORKER_ID}] Final confidence: {location_data.get('confidence_score', 0):.1f}%")
    logger.info(f"[{WORKER_ID}] Final address: {location_data.get('address', 'Unknown')}")
    
    # Ensure we're within target time range
    if total_time < 30:
        sleep_time = 30 - total_time
        logger.info(f"[{WORKER_ID}] Adding {sleep_time:.1f}s delay to meet minimum processing time")
        time.sleep(sleep_time)
    elif total_time > 45:
        logger.warning(f"[{WORKER_ID}] Processing exceeded target time: {total_time:.2f}s > 45s")
    
    return location_data

def process_job_quick(job: JobQueue) -> Dict[str, Any]:
    """
    Process a job in quick analysis mode (faster, less thorough).
    
    Args:
        job: The JobQueue object to process
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"[{WORKER_ID}] Processing job {job.job_id} in QUICK mode")
    
    # Get basic location data without expensive operations
    location_data = get_airbnb_location_data(job.airbnb_url)
    
    # Perform limited visual analysis if photos are available
    if vision_analyzer and location_data.get('photos'):
        try:
            photos_to_analyze = location_data['photos'][:5]  # Only analyze first 5 photos
            logger.info(f"[{WORKER_ID}] Quick visual analysis on {len(photos_to_analyze)} photos")
            
            coords = None
            if location_data.get('latitude') and location_data.get('longitude'):
                coords = (location_data['latitude'], location_data['longitude'])
            
            visual_analysis_results = vision_analyzer(photos_to_analyze, coords)
            
            # Update address only if we have high confidence
            if visual_analysis_results and visual_analysis_results.get('suggested_address'):
                confidence = visual_analysis_results.get('final_address_confidence', 0)
                if confidence > 85:  # Higher threshold for quick mode
                    location_data['address'] = visual_analysis_results['suggested_address']
                    location_data['address_source'] = 'visual_analysis_quick'
                
        except Exception as e:
            logger.warning(f"Skipping visual analysis in quick mode due to error: {e}")
    
    # Add processing metadata
    location_data['processing_mode'] = 'quick'
    location_data['worker_id'] = WORKER_ID
    location_data['processed_at'] = datetime.utcnow().isoformat()
    
    return location_data

def process_single_job(job: JobQueue) -> bool:
    """
    Process a single job from the queue.
    
    Args:
        job: The JobQueue object to process
        
    Returns:
        True if successful, False if failed
    """
    try:
        logger.info(f"[{WORKER_ID}] Starting processing of job {job.job_id} - {job.airbnb_url}")
        start_time = time.time()
        
        # Update job status to processing
        job.last_activity = datetime.utcnow()
        db.session.commit()
        
        # Check if we have a recent cached result
        cached_result = VerificationResult.get_cached(job.airbnb_url, cache_hours=24)
        if cached_result and job.analysis_mode != 'deep':  # Use cache for quick mode only
            logger.info(f"[{WORKER_ID}] Using cached result for {job.airbnb_url}")
            job.mark_completed(cached_result.id)
            return True
        
        # Process based on analysis mode
        if job.analysis_mode == 'deep':
            location_data = process_job_thorough(job)
        else:
            location_data = process_job_quick(job)
        
        # Create verification result
        result = VerificationResult.create_session(
            airbnb_url=job.airbnb_url,
            location_data=location_data
        )
        result.processing_status = 'completed'
        result.processing_completed_at = datetime.utcnow()
        
        db.session.add(result)
        db.session.commit()
        
        # Mark job as completed
        job.mark_completed(result.id)
        
        elapsed_time = time.time() - start_time
        logger.info(f"[{WORKER_ID}] Successfully completed job {job.job_id} in {elapsed_time:.2f} seconds")
        
        # Log some statistics
        address = location_data.get('address', 'Unknown')
        confidence = location_data.get('address_confidence', location_data.get('verification_status', 'Unknown'))
        logger.info(f"[{WORKER_ID}] Result: Address={address}, Confidence={confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"[{WORKER_ID}] Error processing job {job.job_id}: {str(e)}", exc_info=True)
        job.mark_failed(str(e))
        return False

def cleanup_stale_jobs():
    """Mark stale jobs as failed and allow retries."""
    try:
        with app.app_context():
            count = JobQueue.cleanup_stale_jobs(timeout_minutes=STALE_JOB_TIMEOUT)
            if count > 0:
                logger.info(f"[{WORKER_ID}] Cleaned up {count} stale jobs")
    except Exception as e:
        logger.error(f"[{WORKER_ID}] Error cleaning up stale jobs: {e}")

def worker_heartbeat():
    """Update worker status file for monitoring."""
    try:
        status = {
            'worker_id': WORKER_ID,
            'status': 'running',
            'last_heartbeat': datetime.utcnow().isoformat(),
            'pid': os.getpid(),
            'memory_usage': psutil.Process().memory_percent()
        }
        
        # Write status to file
        status_file = f"/tmp/worker_{WORKER_ID}.status"
        with open(status_file, 'w') as f:
            json.dump(status, f)
            
        logger.debug(f"[{WORKER_ID}] Heartbeat updated")
    except Exception as e:
        logger.warning(f"[{WORKER_ID}] Failed to update heartbeat: {e}")

def main():
    """Main worker loop."""
    logger.info(f"[{WORKER_ID}] Background worker starting...")
    logger.info(f"[{WORKER_ID}] PID: {os.getpid()}")
    logger.info(f"[{WORKER_ID}] Rate limiting: {MIN_DELAY}-{MAX_DELAY} seconds between jobs")
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    consecutive_errors = 0
    last_heartbeat = time.time()
    last_cleanup = time.time()
    
    try:
        while not shutdown_requested:
            try:
                # Update heartbeat periodically
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    worker_heartbeat()
                    last_heartbeat = time.time()
                
                # Clean up stale jobs periodically (every 5 minutes)
                if time.time() - last_cleanup > 300:
                    cleanup_stale_jobs()
                    last_cleanup = time.time()
                
                # Check worker health
                if not is_worker_healthy():
                    logger.warning(f"[{WORKER_ID}] Worker health check failed, pausing...")
                    time.sleep(60)
                    continue
                
                # Get next job from queue
                with app.app_context():
                    job = JobQueue.get_next_job(worker_id=WORKER_ID)
                    
                    if not job:
                        logger.debug(f"[{WORKER_ID}] No pending jobs, waiting...")
                        time.sleep(10)  # Wait before checking again
                        continue
                    
                    # Process the job
                    success = process_single_job(job)
                    
                    if success:
                        consecutive_errors = 0
                        # Apply rate limiting
                        delay = get_random_delay()
                        logger.info(f"[{WORKER_ID}] Waiting {delay:.1f} seconds before next job (rate limiting)")
                        time.sleep(delay)
                    else:
                        consecutive_errors += 1
                        # Exponential backoff on errors
                        backoff = min(ERROR_BACKOFF_BASE * (2 ** consecutive_errors), MAX_ERROR_BACKOFF)
                        logger.warning(f"[{WORKER_ID}] Error #{consecutive_errors}, backing off {backoff} seconds")
                        time.sleep(backoff)
                        
                        # If too many consecutive errors, pause longer
                        if consecutive_errors >= 5:
                            logger.error(f"[{WORKER_ID}] Too many consecutive errors, pausing for 5 minutes")
                            time.sleep(300)
                            consecutive_errors = 0
                            
            except KeyboardInterrupt:
                logger.info(f"[{WORKER_ID}] Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"[{WORKER_ID}] Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(30)
                
    finally:
        # Cleanup on shutdown
        logger.info(f"[{WORKER_ID}] Worker shutting down...")
        try:
            # Remove status file
            status_file = f"/tmp/worker_{WORKER_ID}.status"
            if os.path.exists(status_file):
                os.remove(status_file)
        except:
            pass
        logger.info(f"[{WORKER_ID}] Worker stopped")

if __name__ == "__main__":
    main()