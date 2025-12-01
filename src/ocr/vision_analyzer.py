import os
import logging
import json
import base64
import requests
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
from PIL import Image
from functools import lru_cache
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from nlp.address_normalizer import (
    normalize_address, 
    parse_address_components as parse_address,
    standardize_abbreviations,
    extract_addresses_from_text
)
from ocr.google_vision_ocr import analyze_with_google_vision, is_google_vision_available

# Import metrics for tracking OCR and vision operations
from core.metrics import (
    ocr_operations, ocr_processing_time, ocr_confidence,
    images_processed, external_api_calls, external_api_latency
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cache for vision analysis results (24 hour TTL)
_vision_cache = {}
VISION_CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds

def _get_image_hash(image_data: bytes) -> str:
    """Generate a hash for image data to detect duplicates."""
    return hashlib.md5(image_data).hexdigest()

def _resize_image(image_data: bytes, max_width: int = 1024, max_height: int = 1024) -> bytes:
    """
    Resize image to reduce payload size while maintaining aspect ratio.
    
    Args:
        image_data: Raw image bytes
        max_width: Maximum width in pixels (default 1024)
        max_height: Maximum height in pixels (default 1024)
    
    Returns:
        Resized image bytes
    """
    try:
        img = Image.open(BytesIO(image_data))
        
        # Convert RGBA to RGB if needed (for JPEG conversion)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img
        
        # Calculate new dimensions while maintaining aspect ratio
        ratio = min(max_width/img.width, max_height/img.height)
        if ratio < 1:  # Only resize if image is larger than max dimensions
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {img.width}x{img.height} to {new_width}x{new_height}")
        
        # Save to bytes with optimized JPEG
        output = BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        return output.getvalue()
    except Exception as e:
        logger.warning(f"Could not resize image: {e}")
        return image_data  # Return original if resize fails

def encode_image_from_url(image_url: str, resize: bool = True) -> Optional[Tuple[str, str]]:
    """
    Download, optionally resize, and encode an image from URL to base64.
    
    Returns:
        Tuple of (base64_encoded_image, image_hash) or None if failed
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_data = response.content
        
        # Generate hash for duplicate detection
        image_hash = _get_image_hash(image_data)
        
        # Resize image if requested to reduce payload
        if resize:
            image_data = _resize_image(image_data)
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image, image_hash
    except Exception as e:
        logger.error(f"Error encoding image from {image_url}: {str(e)}")
        return None

def _get_cache_key(photo_urls: List[str], known_address: Optional[str]) -> str:
    """Generate a cache key for vision analysis results."""
    urls_str = '|'.join(sorted(photo_urls[:5]))  # Only consider first 5 URLs for cache
    address_str = known_address or 'no-address'
    return hashlib.sha256(f"{urls_str}:{address_str}".encode()).hexdigest()

def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached vision analysis result if available and not expired."""
    if cache_key in _vision_cache:
        cached_data, timestamp = _vision_cache[cache_key]
        if time.time() - timestamp < VISION_CACHE_TTL:
            logger.info("Returning cached vision analysis result")
            return cached_data
        else:
            # Remove expired cache
            del _vision_cache[cache_key]
    return None

def _cache_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Cache vision analysis result."""
    _vision_cache[cache_key] = (result, time.time())
    # Clean up old cache entries (keep max 100 entries)
    if len(_vision_cache) > 100:
        # Remove oldest entries
        sorted_entries = sorted(_vision_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_entries[:20]:  # Remove 20 oldest
            del _vision_cache[old_key]

def _get_empty_vision_result() -> Dict[str, Any]:
    """Return an empty vision analysis result structure."""
    return {
        "visual_address_clues": [],
        "detected_text": [],
        "landmarks": [],
        "building_features": [],
        "ocr_address_data": {
            "house_numbers": [],
            "street_signs": [],
            "mailbox_text": [],
            "building_markers": [],
            "confidence_scores": {
                "house_numbers": 0,
                "street_signs": 0,
                "mailbox_text": 0,
                "building_markers": 0
            }
        },
        "property_features": {
            "bedrooms": None,
            "bathrooms": None,
            "property_type": None,
            "style": None,
            "special_features": [],
            "floor_type": None,
            "kitchen_features": [],
            "view": None,
            "stories": None,
            "exterior": None
        },
        "estimated_location": None,
        "confidence_score": 0
    }

def _convert_google_vision_results(google_results: Dict[str, Any], known_address: Optional[str] = None) -> Dict[str, Any]:
    """Convert Google Vision OCR results to our standard format."""
    result = _get_empty_vision_result()
    
    # Extract house numbers from Google Vision results
    if google_results.get('all_house_numbers'):
        for house_num in google_results['all_house_numbers']:
            result['ocr_address_data']['house_numbers'].append({
                'text': house_num,
                'confidence': 80,  # Google Vision generally has good confidence
                'location': 'google_vision_ocr'
            })
        result['ocr_address_data']['confidence_scores']['house_numbers'] = 80
        
    # Extract all detected text
    if google_results.get('photo_results'):
        all_text = []
        for photo_result in google_results['photo_results']:
            if photo_result.get('all_text'):
                for text_item in photo_result['all_text'][:10]:  # Limit to top 10 per photo
                    all_text.append(text_item.get('text', ''))
        result['detected_text'] = all_text[:30]  # Limit total to 30 items
        
    # Get best house number candidate
    if google_results.get('best_house_number'):
        best = google_results['best_house_number']
        result['estimated_location'] = f"{best['text']} {known_address.split(',')[-1] if known_address else ''}"
        result['confidence_score'] = best.get('confidence', 0)
        
    # Add summary information
    if google_results.get('summary'):
        summary = google_results['summary']
        result['visual_address_clues'].append(f"Found {summary.get('house_numbers_found', 0)} house numbers in {summary.get('photos_analyzed', 0)} photos")
        
    return result

def _prepare_gpt4_messages(photo_urls: List[str], known_address: Optional[str] = None) -> List[ChatCompletionMessageParam]:
    """Prepare messages for GPT-4 Vision API call with enhanced landmark and feature extraction."""
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """You are an expert vision analyst specializing in real estate photo analysis, OCR, and landmark identification.
            Your task is to perform COMPREHENSIVE VISUAL ANALYSIS extracting address information, landmarks, and building features.
            
            Return STRUCTURED JSON with comprehensive analysis:
            1. OCR Results - House numbers, street signs, mailbox text, building markers
            2. Landmarks - Nearby businesses, notable buildings, transit stops, chain stores
            3. Building Features - Architecture, materials, colors, unique elements
            4. Environmental Context - Neighborhood type, street characteristics, geographic features
            5. Property Features - Type, style, special features, exterior details
            
            For landmarks, provide:
            - Type (business/landmark/transit/etc)
            - Name if visible
            - Detailed description
            - Relative position to property
            - Confidence score
            
            For building features, provide:
            - Feature type (architecture/material/color/unique_element)
            - Detailed description  
            - Distinctive level (highly/moderately/common)
            - Specific identification details
            
            Provide confidence scores (0-100) for ALL extractions."""
        }
    ]
    
    # Add user message with context
    user_content = []
    if known_address:
        user_content.append({
            "type": "text",
            "text": f"""Analyze these photos from a property allegedly at: {known_address}
            
            Extract:
            1. ALL visible text (house numbers, street signs, business names, etc)
            2. ALL nearby landmarks and businesses with names
            3. ALL distinctive building features and architectural elements
            4. Environmental and neighborhood context
            
            Verify if visual evidence matches the alleged address."""
        })
    else:
        user_content.append({
            "type": "text",
            "text": """Perform COMPREHENSIVE visual analysis:
            
            1. Extract ALL visible text (house numbers, street signs, business names)
            2. Identify ALL landmarks, businesses, and notable structures
            3. Describe ALL building features and architectural elements
            4. Capture environmental and neighborhood context
            
            Be DETAILED in landmark and feature identification for address verification."""
        })
    
    # Add images
    for url in photo_urls:
        result = encode_image_from_url(url, resize=True)
        if result:
            base64_image, _ = result
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })
    
    messages.append({"role": "user", "content": user_content})
    return messages

def _process_gpt4_response(response) -> Dict[str, Any]:
    """Process the GPT-4 Vision API response with enhanced landmark and building feature handling."""
    try:
        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)
        
        # Ensure all required fields are present with proper structure
        if 'ocr_address_data' not in result_data:
            result_data['ocr_address_data'] = {}
        if 'visual_address_clues' not in result_data:
            result_data['visual_address_clues'] = []
        if 'confidence_score' not in result_data:
            result_data['confidence_score'] = 0
            
        # Ensure landmarks array exists and is properly structured
        if 'landmarks' not in result_data or not isinstance(result_data['landmarks'], list):
            result_data['landmarks'] = []
        else:
            # Process and validate landmarks
            processed_landmarks = []
            for landmark in result_data['landmarks']:
                if isinstance(landmark, dict):
                    processed_landmarks.append({
                        'type': landmark.get('type', 'unknown'),
                        'name': landmark.get('name', ''),
                        'description': landmark.get('description', ''),
                        'relative_position': landmark.get('relative_position', ''),
                        'confidence': landmark.get('confidence', 0)
                    })
                elif isinstance(landmark, str):
                    # Convert simple string to structured format
                    processed_landmarks.append({
                        'type': 'general',
                        'name': '',
                        'description': landmark,
                        'relative_position': 'nearby',
                        'confidence': 50
                    })
            result_data['landmarks'] = processed_landmarks
            
        # Ensure building_features array exists and is properly structured
        if 'building_features' not in result_data or not isinstance(result_data['building_features'], list):
            result_data['building_features'] = []
        else:
            # Process and validate building features
            processed_features = []
            for feature in result_data['building_features']:
                if isinstance(feature, dict):
                    processed_features.append({
                        'feature_type': feature.get('feature_type', 'general'),
                        'description': feature.get('description', ''),
                        'distinctive_level': feature.get('distinctive_level', 'common'),
                        'details': feature.get('details', '')
                    })
                elif isinstance(feature, str):
                    # Convert simple string to structured format
                    processed_features.append({
                        'feature_type': 'general',
                        'description': feature,
                        'distinctive_level': 'moderately_distinctive',
                        'details': ''
                    })
            result_data['building_features'] = processed_features
            
        # Process environmental context if present
        if 'environmental_context' in result_data and isinstance(result_data['environmental_context'], dict):
            # Ensure environmental context has proper structure
            env_context = result_data['environmental_context']
            result_data['environmental_context'] = {
                'neighborhood_type': env_context.get('neighborhood_type', ''),
                'street_characteristics': env_context.get('street_characteristics', ''),
                'vegetation': env_context.get('vegetation', ''),
                'topography': env_context.get('topography', '')
            }
            
        # Log summary of extracted landmarks and features
        landmark_count = len(result_data['landmarks'])
        feature_count = len(result_data['building_features'])
        if landmark_count > 0 or feature_count > 0:
            logger.info(f"Extracted {landmark_count} landmarks and {feature_count} building features")
            
        return result_data
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.error(f"Error processing GPT-4 response: {str(e)}")
        return _get_empty_vision_result()

def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR text to handle common OCR errors and standardize formats.
    Uses the address_normalizer module for better standardization.
    
    Args:
        text: Raw OCR text that may contain errors
        
    Returns:
        Normalized and cleaned text
    """
    if not text:
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Apply OCR-specific corrections for common misreadings
    # Only apply to text that appears to be a house number
    if re.match(r'^[lIOSZGB0-9]{1,5}[A-Za-z]?\b', text, re.IGNORECASE):
        # This looks like a house number - apply aggressive OCR corrections
        number_match = re.match(r'^([lIOSZGB0-9]{1,5}[A-Za-z]?)', text, re.IGNORECASE)
        if number_match:
            number_part = number_match.group(1)
            # Apply corrections to the number part only
            corrected = number_part.replace('l', '1').replace('I', '1')  # l/I to 1
            corrected = corrected.replace('O', '0').replace('o', '0')  # O/o to 0
            corrected = corrected.replace('S', '5').replace('s', '5')  # S/s to 5  
            corrected = corrected.replace('Z', '2').replace('z', '2')  # Z/z to 2
            corrected = corrected.replace('G', '6').replace('g', '6')  # G/g to 6
            corrected = corrected.replace('B', '8').replace('b', '8')  # B/b to 8
            # Replace the number part in the original text
            text = text.replace(number_part, corrected, 1)
    
    # Use the address normalizer for standardization
    # This handles abbreviations, capitalization, and formatting
    normalized = normalize_address(text)
    
    return normalized

def extract_address_components(texts: List[str]) -> Dict[str, Any]:
    """
    Extract and structure address components from a list of OCR texts.
    Uses the address_normalizer module for better parsing.
    
    Args:
        texts: List of detected text strings
        
    Returns:
        Dictionary with structured address components
    """
    components = {
        "house_numbers": [],
        "street_names": [],
        "apartment_units": [],
        "postal_codes": [],
        "building_names": [],
        "suggested_address": None
    }
    
    # First, try to extract full addresses from the text
    full_text = ' '.join(texts)
    potential_addresses = extract_addresses_from_text(full_text)
    
    if potential_addresses and potential_addresses[0]['confidence'] > 70:
        # Use the best detected address
        components["suggested_address"] = potential_addresses[0]['address']
        # Parse it for components
        parsed = parse_address(potential_addresses[0]['address'])
        if parsed['house_number']:
            components["house_numbers"].append(parsed['house_number'])
            logger.debug(f"Added house number from full address: {parsed['house_number']}")
        if parsed['street_name']:
            street_full = parsed['street_name']
            if parsed['street_type']:
                street_full += f" {parsed['street_type']}"
            components["street_names"].append(street_full)
        if parsed['unit']:
            components["apartment_units"].append(parsed['unit'])
        if parsed['postal_code']:
            components["postal_codes"].append(parsed['postal_code'])
    
    # Also process individual text snippets
    for text in texts:
        if not text:
            continue
            
        # Normalize the text
        normalized = normalize_ocr_text(text)
        logger.debug(f"Processing text snippet: '{text}' -> normalized: '{normalized}'")
        
        # CRITICAL FIX: Check if this is a standalone number (house number)
        # This handles cases like "109" or "123A" that would otherwise be misclassified
        standalone_number_pattern = r'^(\d{1,5}[A-Za-z]?)$'
        number_match = re.match(standalone_number_pattern, normalized.strip())
        if number_match:
            # This is a standalone number - it's a house number!
            house_num = number_match.group(1)
            if house_num not in components["house_numbers"]:
                components["house_numbers"].append(house_num)
                logger.info(f"Detected standalone house number: '{house_num}' from text: '{text}'")
            # Skip further parsing for this text since we've identified it as a house number
            continue
        
        # Parse the normalized text for more complex patterns
        parsed = parse_address(normalized)
        logger.debug(f"Parse result for '{normalized}': {parsed}")
        
        # Extract components
        if parsed['house_number'] and parsed['house_number'] not in components["house_numbers"]:
            components["house_numbers"].append(parsed['house_number'])
            logger.debug(f"Added house number from parsing: {parsed['house_number']}")
        
        # Only add as street name if it's not just a number
        if parsed['street_name']:
            # Check if the street name is just a number (which would be a misclassified house number)
            if not re.match(r'^\d{1,5}[A-Za-z]?$', parsed['street_name'].strip()):
                street_full = parsed['street_name']
                if parsed['street_type']:
                    street_full += f" {parsed['street_type']}"
                if street_full not in components["street_names"]:
                    components["street_names"].append(street_full)
                    logger.debug(f"Added street name: {street_full}")
            else:
                # This "street name" is actually a house number
                if parsed['street_name'] not in components["house_numbers"]:
                    components["house_numbers"].append(parsed['street_name'])
                    logger.info(f"Reclassified street name '{parsed['street_name']}' as house number")
        
        if parsed['unit'] and parsed['unit'] not in components["apartment_units"]:
            components["apartment_units"].append(parsed['unit'])
            logger.debug(f"Added apartment unit: {parsed['unit']}")
        
        if parsed['postal_code'] and parsed['postal_code'] not in components["postal_codes"]:
            components["postal_codes"].append(parsed['postal_code'])
            logger.debug(f"Added postal code: {parsed['postal_code']}")
        
        # Building name (longer text that doesn't match other patterns)
        if len(normalized) > 10 and not any([
            parsed['house_number'],
            parsed['street_name'],
            parsed['unit'],
            parsed['postal_code'],
            re.match(r'^\d{1,5}[A-Za-z]?$', normalized.strip())  # Not a standalone number
        ]):
            # Likely a building or complex name
            if normalized not in components["building_names"]:
                components["building_names"].append(normalized)
                logger.debug(f"Added building name: {normalized}")
    
    # Remove duplicates while preserving order
    for key in components:
        if key != "suggested_address" and isinstance(components[key], list):
            components[key] = list(dict.fromkeys(components[key]))
    
    logger.info(f"Structured address extraction: {components}")
    
    return components

def analyze_with_gpt4_vision(photo_urls: List[str], known_address: Optional[str] = None, 
                           timeout: Optional[int] = None, max_photos: int = 5) -> Dict[str, Any]:
    """
    Analyze photos using GPT-4 Vision with configurable timeout.
    
    Args:
        photo_urls: List of photo URLs to analyze
        known_address: Known address to help with verification (optional)
        timeout: Timeout in seconds for the API call (None for no timeout)
        max_photos: Maximum number of photos to analyze
    
    Returns:
        Dictionary containing extracted location information
    
    Raises:
        TimeoutError: If the API call exceeds the timeout
    """
    # Limit photos for optimization
    photo_urls = photo_urls[:max_photos] if len(photo_urls) > max_photos else photo_urls
    
    # Check cache first
    cache_key = _get_cache_key(photo_urls, known_address)
    cached_result = _get_cached_result(cache_key)
    if cached_result:
        return cached_result
    
    if not photo_urls:
        return _get_empty_vision_result()
    
    # Start tracking OCR processing time
    start_time = time.time()
    
    try:
        # Track images being processed
        images_processed.labels(processing_type='gpt4_vision').inc(len(photo_urls))
        
        # [Rest of the GPT-4 Vision analysis logic will be moved here from analyze_property_photos]
        # This is the core GPT-4 analysis with timeout support
        messages = _prepare_gpt4_messages(photo_urls, known_address)
        
        # Track API call start time
        api_start = time.time()
        
        # Make the API call with timeout
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.3,
            timeout=timeout  # Apply timeout to the API call
        )
        
        # Track API latency
        api_latency = time.time() - api_start
        external_api_latency.labels(service='openai', endpoint='gpt4_vision').observe(api_latency)
        external_api_calls.labels(service='openai', endpoint='gpt4_vision', status='success').inc()
        
        # Process and return the response
        result = _process_gpt4_response(response)
        
        # Track OCR metrics
        processing_time = time.time() - start_time
        ocr_processing_time.labels(provider='gpt4_vision').observe(processing_time)
        ocr_operations.labels(provider='gpt4_vision', status='success').inc()
        
        # Track confidence if available
        if result.get('confidence_score'):
            ocr_confidence.labels(provider='gpt4_vision').set(result['confidence_score'])
        
        _cache_result(cache_key, result)
        return result
        
    except Exception as e:
        # Track processing time and failure
        processing_time = time.time() - start_time
        ocr_processing_time.labels(provider='gpt4_vision').observe(processing_time)
        ocr_operations.labels(provider='gpt4_vision', status='failure').inc()
        external_api_calls.labels(service='openai', endpoint='gpt4_vision', status='error').inc()
        
        if "timeout" in str(e).lower():
            raise TimeoutError(f"GPT-4 Vision timed out after {timeout} seconds")
        raise

def analyze_with_google_vision_fallback(photo_urls: List[str], known_address: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze photos using Google Vision as a fallback when GPT-4 times out.
    
    Args:
        photo_urls: List of photo URLs to analyze
        known_address: Known address to help with verification (optional)
    
    Returns:
        Dictionary containing extracted location information from Google Vision
    """
    logger.info("Using Google Vision as fallback for photo analysis")
    
    if not is_google_vision_available():
        logger.warning("Google Vision is not available as fallback")
        return _get_empty_vision_result()
    
    # Start tracking processing time
    start_time = time.time()
    
    try:
        # Track images being processed
        images_processed.labels(processing_type='google_vision').inc(len(photo_urls))
        
        # Use Google Vision OCR
        google_results = analyze_with_google_vision(photo_urls)
        
        # Convert Google Vision results to our standard format
        result = _convert_google_vision_results(google_results, known_address)
        result['analysis_source'] = 'google_vision_fallback'
        
        # Track OCR metrics
        processing_time = time.time() - start_time
        ocr_processing_time.labels(provider='google_vision').observe(processing_time)
        ocr_operations.labels(provider='google_vision', status='success').inc()
        
        # Track confidence if available
        if result.get('confidence_score'):
            ocr_confidence.labels(provider='google_vision').set(result['confidence_score'])
        
        return result
        
    except Exception as e:
        # Track failure metrics
        processing_time = time.time() - start_time
        ocr_processing_time.labels(provider='google_vision').observe(processing_time)
        ocr_operations.labels(provider='google_vision', status='failure').inc()
        
        logger.error(f"Google Vision fallback failed: {str(e)}")
        return _get_empty_vision_result()

def analyze_property_photos(photo_urls: List[str], known_address: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze photos using staged timeout configuration with graceful fallback.
    Implements progressive loading with quick verification first, then detailed analysis.
    
    Args:
        photo_urls: List of photo URLs to analyze
        known_address: Known address to help with verification (optional)
        
    Returns:
        Dictionary containing extracted location information and property features
    """
    # STAGED TIMEOUT CONFIGURATION
    # Stage 1: Quick verification (5 seconds) - 2 photos only
    # Stage 2: Detailed analysis (30 seconds) - 5 photos
    # Stage 3: Full analysis (unlimited in background) - all photos
    
    logger.info("Starting staged photo analysis with progressive loading")
    
    # Stage 1: Quick verification with 5-second timeout
    try:
        logger.info("Stage 1: Quick verification (5s timeout, 2 photos)")
        quick_result = analyze_with_gpt4_vision(
            photo_urls[:2], 
            known_address, 
            timeout=5,
            max_photos=2
        )
        quick_result['analysis_stage'] = 'quick'
        quick_result['analysis_source'] = 'gpt4_vision'
        logger.info("Quick verification completed successfully")
        return quick_result
        
    except TimeoutError:
        logger.warning("GPT-4 Vision timed out in quick mode (5s), trying detailed mode")
        
        # Stage 2: Detailed analysis with 30-second timeout
        try:
            logger.info("Stage 2: Detailed analysis (30s timeout, 5 photos)")
            detailed_result = analyze_with_gpt4_vision(
                photo_urls[:5],
                known_address,
                timeout=30,
                max_photos=5
            )
            detailed_result['analysis_stage'] = 'detailed'
            detailed_result['analysis_source'] = 'gpt4_vision'
            logger.info("Detailed analysis completed successfully")
            return detailed_result
            
        except TimeoutError:
            logger.warning("GPT-4 Vision timed out in detailed mode (30s), falling back to Google Vision")
            
            # Stage 3: Graceful fallback to Google Vision
            try:
                logger.info("Stage 3: Falling back to Google Vision OCR")
                fallback_result = analyze_with_google_vision_fallback(photo_urls, known_address)
                fallback_result['analysis_stage'] = 'fallback'
                logger.info("Google Vision fallback completed successfully")
                return fallback_result
                
            except Exception as e:
                logger.error(f"All analysis methods failed: {str(e)}")
                empty_result = _get_empty_vision_result()
                empty_result['analysis_stage'] = 'failed'
                empty_result['error'] = str(e)
                return empty_result
    
    except Exception as e:
        logger.error(f"Unexpected error in staged analysis: {str(e)}")
        # Try fallback on any error
        try:
            fallback_result = analyze_with_google_vision_fallback(photo_urls, known_address)
            fallback_result['analysis_stage'] = 'error_fallback'
            return fallback_result
        except:
            empty_result = _get_empty_vision_result()
            empty_result['analysis_stage'] = 'failed'
            empty_result['error'] = str(e)
            return empty_result
    
    if not photo_urls:
        logger.warning("No photos provided for analysis")
        return {
            "visual_address_clues": [],
            "detected_text": [],
            "landmarks": [],
            "building_features": [],
            "ocr_address_data": {
                "house_numbers": [],
                "street_signs": [],
                "mailbox_text": [],
                "building_markers": [],
                "confidence_scores": {
                    "house_numbers": 0,
                    "street_signs": 0,
                    "mailbox_text": 0,
                    "building_markers": 0
                }
            },
            "property_features": {
                "bedrooms": None,
                "bathrooms": None,
                "property_type": None,
                "style": None,
                "special_features": [],
                "floor_type": None,
                "kitchen_features": [],
                "view": None,
                "stories": None,
                "exterior": None
            },
            "estimated_location": None,
            "confidence_score": 0
        }
    
    try:
        # Prepare messages for GPT-4 Vision
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": """You are an expert vision analyst specializing in real estate photo analysis, OCR, and landmark identification.
                Your task is to perform COMPREHENSIVE VISUAL ANALYSIS extracting address information, landmarks, and building features.
                
                PART 1: CRITICAL OCR TARGETS - Read ALL text from EVERY photo:
                
                1. **HOUSE NUMBERS** (HIGHEST PRIORITY):
                   - CAREFULLY SCAN each photo for numbers on building exteriors
                   - Look for numbers on building facades, near doors, on walls, on siding
                   - Check for numbers on pillars, gates, fence posts, or entrance posts
                   - Read numbers from address plaques, signs, or painted/mounted directly on building
                   - Note: Could be 1-5 digits (like "130", "141", "151"), may include letters (e.g., "123A")
                
                2. **STREET SIGNS** (HIGH PRIORITY):
                   - Read street name signs visible in any photo
                   - Look for intersection signs showing multiple streets
                   - Check for directional indicators (N, S, E, W)
                   - Read the COMPLETE street name including type (Street, Avenue, Road, etc.)
                
                3. **MAILBOX TEXT** (HIGH PRIORITY):
                   - Read any numbers or text on mailboxes
                   - Check for apartment/unit numbers on mailbox banks
                   - Look for names that might indicate building or complex
                   - Read any address labels or stickers
                
                PART 2: LANDMARKS AND VISUAL CONTEXT - Identify ALL visible landmarks:
                
                1. **NEARBY LANDMARKS** (CRITICAL FOR DEEP ANALYSIS):
                   - Identify nearby businesses (stores, restaurants, banks, gas stations)
                   - Note any churches, schools, hospitals, government buildings
                   - Identify parks, monuments, bridges, or notable structures
                   - Read business signs and store names visible in background
                   - Note distinctive neighboring buildings (color, style, unique features)
                   - Identify any transit stops (bus stops, subway entrances) with names
                   - Note any visible corporate logos or chain stores
                
                2. **BUILDING FEATURES** (ESSENTIAL FOR PROPERTY MATCHING):
                   - Architecture style (Victorian, Modern, Colonial, Ranch, Tudor, etc.)
                   - Building material (brick, stucco, wood siding, stone, concrete)
                   - Building color and color combinations (be specific)
                   - Unique architectural elements (turrets, bay windows, dormers, columns)
                   - Roof type and material (shingle, tile, metal, flat)
                   - Window style and configuration (bay windows, French windows, unique patterns)
                   - Door style and color (double door, glass panels, unique design)
                   - Fence or gate style if present (iron, wood, stone wall)
                   - Driveway or parking arrangement
                   - Landscaping features (distinctive trees, garden features)
                   - Any unique decorative elements or artwork
                
                3. **VISUAL CLUES AND CONTEXT**:
                   - Street furniture (lamp posts style, benches, trash cans)
                   - Traffic signs beyond street names (stop signs, yield, parking signs)
                   - Utility features (fire hydrants color/style, power lines configuration)
                   - Geographic features (hills, water bodies, mountains in background)
                   - Weather/seasonal indicators that might help with location
                   - Any construction or renovation indicators
                   - Neighboring property characteristics
                
                Return COMPREHENSIVE JSON with ALL visual analysis:
                {
                    "ocr_address_data": {
                        "house_numbers": [
                            {"text": "extracted_number", "confidence": 0-100, "location": "where_found"}
                        ],
                        "street_signs": [
                            {"text": "street_name", "confidence": 0-100, "type": "Street/Ave/Rd/etc"}
                        ],
                        "mailbox_text": [
                            {"text": "mailbox_content", "confidence": 0-100, "details": "any_context"}
                        ],
                        "building_markers": [
                            {"text": "building_name_or_plaque", "confidence": 0-100, "type": "sign/plaque/marker"}
                        ],
                        "confidence_scores": {
                            "house_numbers": overall_confidence_0-100,
                            "street_signs": overall_confidence_0-100,
                            "mailbox_text": overall_confidence_0-100,
                            "building_markers": overall_confidence_0-100
                        }
                    },
                    "visual_address_clues": ["all specific address indicators found"],
                    "detected_text": ["ALL text visible in images, even non-address text"],
                    "landmarks": [
                        {
                            "type": "business/landmark/transit/etc",
                            "name": "specific name if visible",
                            "description": "detailed description",
                            "relative_position": "where relative to property",
                            "confidence": 0-100
                        }
                    ],
                    "building_features": [
                        {
                            "feature_type": "architecture/material/color/unique_element",
                            "description": "detailed description",
                            "distinctive_level": "highly_distinctive/moderately_distinctive/common",
                            "details": "specific details that help identification"
                        }
                    ],
                    "property_features": {
                        "bedrooms": number_or_null,
                        "bathrooms": number_or_null,
                        "property_type": "house/condo/apartment/etc or null",
                        "style": "detailed_architectural_style",
                        "special_features": ["list of special features"],
                        "floor_type": "type or null",
                        "kitchen_features": ["list"],
                        "view": "type or null",
                        "stories": number_or_null,
                        "exterior": "detailed exterior description"
                    },
                    "environmental_context": {
                        "neighborhood_type": "urban/suburban/rural",
                        "street_characteristics": "busy/quiet, wide/narrow, etc",
                        "vegetation": "types of trees or plants if distinctive",
                        "topography": "flat/hilly/etc"
                    },
                    "estimated_location": "best guess of specific address if possible",
                    "confidence_score": 0-100
                }
                
                Provide COMPREHENSIVE visual analysis including ALL landmarks and building features for property identification."""
            }
        ]
        
        # Add user message with context
        user_content: List[Union[Dict[str, str], Dict[str, Any]]] = []
        if known_address:
            user_content.append({
                "type": "text",
                "text": f"""These photos are from a property allegedly at or near: {known_address}.
                
                COMPREHENSIVE ANALYSIS REQUIRED:
                
                1. OCR EXTRACTION (Critical):
                   - Read ALL house numbers visible on buildings, doors, or gates
                   - Extract ALL street signs with complete street names
                   - Read ALL mailbox text including numbers and labels
                   - Capture text from building plaques, markers, or entrance signs
                   - Extract ANY other text that could indicate the address
                
                2. LANDMARK IDENTIFICATION (Essential for Deep Analysis):
                   - Identify ALL nearby businesses with their names and signs
                   - Note churches, schools, hospitals, or government buildings
                   - Identify any parks, monuments, or notable structures
                   - Read all visible store/restaurant names in the area
                   - Note distinctive neighboring buildings and their features
                   - Identify transit stops with their names/numbers
                
                3. BUILDING FEATURE EXTRACTION (Critical for Matching):
                   - Describe the architecture style in detail
                   - Identify building materials and colors (be specific)
                   - Note unique architectural elements (windows, doors, roof, etc)
                   - Describe landscaping and exterior features
                   - Note any distinctive decorative elements
                   - Identify fence/gate style and materials
                
                4. ENVIRONMENTAL CONTEXT:
                   - Describe the neighborhood type and street characteristics
                   - Note geographic features (hills, water, mountains)
                   - Identify street furniture and utility features
                   - Note vegetation types if distinctive
                
                Verify if the visual evidence matches the alleged address: {known_address}
                Provide confidence scores (0-100) for each element."""
            })
        else:
            user_content.append({
                "type": "text", 
                "text": """Perform COMPREHENSIVE visual analysis on these property photos:
                
                1. OCR EXTRACTION (Primary):
                   - Read ALL house numbers from buildings, doors, gates, or address plaques
                   - Extract ALL street names from any visible street signs
                   - Read ALL text from mailboxes including numbers and labels
                   - Extract text from ALL building markers, plaques, or entrance signs
                   - Capture ANY other visible text that could indicate location
                
                2. LANDMARK IDENTIFICATION (Critical):
                   - Identify ALL nearby businesses, stores, restaurants with names
                   - Note any churches, schools, hospitals, government buildings
                   - Identify parks, monuments, bridges, or notable structures
                   - Read all business signs visible in any photo
                   - Note distinctive neighboring buildings (color, style, features)
                   - Identify transit stops with names/numbers
                   - Note any chain stores or recognizable logos
                
                3. BUILDING FEATURES (Essential):
                   - Architecture style (Victorian, Modern, Colonial, Ranch, etc)
                   - Building materials (brick, stucco, wood, stone, etc)
                   - Building colors and color combinations
                   - Unique architectural elements (turrets, bay windows, columns)
                   - Roof type and material
                   - Window and door styles
                   - Fence or gate details
                   - Landscaping and garden features
                   - Any unique or distinctive decorative elements
                
                4. VISUAL CONTEXT:
                   - Neighborhood type (urban/suburban/rural)
                   - Street characteristics (wide/narrow, busy/quiet)
                   - Geographic features visible
                   - Utility and street furniture
                   - Vegetation and topography
                
                Provide confidence scores (0-100) for each extraction.
                Be DETAILED and COMPREHENSIVE in identifying landmarks and features."""
            })
        
        # Process images with optimizations
        analyzed_count = 0
        seen_hashes = set()
        image_data_list = []
        MAX_IMAGES = 3  # Reduced to avoid timeouts while still capturing key details
        
        # First, collect and deduplicate images
        for i, url in enumerate(photo_urls[:5]):  # Check first 5 for duplicates but only use 3
            try:
                # Download and resize image
                result = encode_image_from_url(url, resize=True)
                if not result:
                    continue
                    
                base64_image, image_hash = result
                
                # Skip duplicate images
                if image_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate image: {url}")
                    continue
                
                seen_hashes.add(image_hash)
                
                # Determine detail level based on image position and content
                # Use high detail for first 2 images (likely main property photos)
                # and low detail for others to save API costs
                detail_level = "high" if i < 2 else "low"
                
                image_data_list.append({
                    "url": url,
                    "base64": base64_image,
                    "detail": detail_level
                })
                
                # Stop after we have MAX_IMAGES unique images
                if len(image_data_list) >= MAX_IMAGES:
                    break
                    
            except Exception as e:
                logger.warning(f"Could not process image {url}: {str(e)}")
                continue
        
        # Add processed images to the message
        for img_data in image_data_list:
            # Use data URL instead of remote URL for better reliability
            data_url = f"data:image/jpeg;base64,{img_data['base64']}"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": img_data['detail']
                }
            })
            analyzed_count += 1
            logger.debug(f"Added image with {img_data['detail']} detail level")
        
        if analyzed_count == 0:
            logger.error("No images could be prepared for analysis")
            return {
                "visual_address_clues": [],
                "detected_text": [],
                "landmarks": [],
                "building_features": [],
                "ocr_address_data": {
                    "house_numbers": [],
                    "street_signs": [],
                    "mailbox_text": [],
                    "building_markers": [],
                    "confidence_scores": {
                        "house_numbers": 0,
                        "street_signs": 0,
                        "mailbox_text": 0,
                        "building_markers": 0
                    }
                },
                "property_features": {
                    "bedrooms": None,
                    "bathrooms": None,
                    "property_type": None,
                    "style": None,
                    "special_features": [],
                    "floor_type": None,
                    "kitchen_features": [],
                    "view": None,
                    "stories": None,
                    "exterior": None
                },
                "estimated_location": None,
                "confidence_score": 0
            }
        
        messages.append({
            "role": "user",
            "content": user_content  # type: ignore
        })
        
        # Call GPT-4 Vision
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # GPT-4o has vision capabilities
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse response
        message_content = response.choices[0].message.content
        if message_content is None:
            logger.error("No content in OpenAI response")
            return {
                "visual_address_clues": [],
                "detected_text": [],
                "landmarks": [],
                "building_features": [],
                "ocr_address_data": {
                    "house_numbers": [],
                    "street_signs": [],
                    "mailbox_text": [],
                    "building_markers": [],
                    "confidence_scores": {
                        "house_numbers": 0,
                        "street_signs": 0,
                        "mailbox_text": 0,
                        "building_markers": 0
                    }
                },
                "property_features": {
                    "bedrooms": None,
                    "bathrooms": None,
                    "property_type": None,
                    "style": None,
                    "special_features": [],
                    "floor_type": None,
                    "kitchen_features": [],
                    "view": None,
                    "stories": None,
                    "exterior": None
                },
                "estimated_location": None,
                "confidence_score": 0
            }
        analysis = json.loads(message_content)
        
        logger.info(f"Successfully analyzed {analyzed_count} photos")
        logger.info(f"Found {len(analysis.get('visual_address_clues', []))} address clues")
        
        # Process OCR data with text normalization
        if 'ocr_address_data' in analysis:
            ocr_data = analysis['ocr_address_data']
            
            # Normalize and extract house numbers
            if 'house_numbers' in ocr_data and isinstance(ocr_data['house_numbers'], list):
                for item in ocr_data['house_numbers']:
                    if isinstance(item, dict) and 'text' in item:
                        # Extract just the number from text like "house number 109 on front facade"
                        text = str(item['text']).lower()
                        # Try to find a house number pattern
                        number_patterns = [
                            r'house\s*number\s*(\d+[a-z]?)',  # "house number 109"
                            r'number\s*(\d+[a-z]?)',           # "number 109"
                            r'(\d+[a-z]?)\s*on\s*front',       # "109 on front"
                            r'^\s*(\d+[a-z]?)\s*$',           # just "109"
                            r'(\d+[a-z]?)'                     # any number
                        ]
                        extracted_number = None
                        for pattern in number_patterns:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                extracted_number = match.group(1)
                                break
                        
                        # Store both the original and extracted
                        item['original_text'] = item['text']
                        if extracted_number:
                            item['text'] = extracted_number  # Replace with just the number
                            item['normalized'] = normalize_ocr_text(extracted_number)
                        else:
                            item['normalized'] = normalize_ocr_text(item['text'])
                        
                        logger.debug(f"House number extraction: '{item.get('original_text')}' -> '{item['text']}'")
                        
                logger.info(f"OCR detected {len(ocr_data.get('house_numbers', []))} house numbers")
            
            # Normalize street signs
            if 'street_signs' in ocr_data and isinstance(ocr_data['street_signs'], list):
                for item in ocr_data['street_signs']:
                    if isinstance(item, dict) and 'text' in item:
                        item['normalized'] = normalize_ocr_text(item['text'])
                logger.info(f"OCR detected {len(ocr_data.get('street_signs', []))} street signs")
            
            # Normalize mailbox text
            if 'mailbox_text' in ocr_data and isinstance(ocr_data['mailbox_text'], list):
                for item in ocr_data['mailbox_text']:
                    if isinstance(item, dict) and 'text' in item:
                        item['normalized'] = normalize_ocr_text(item['text'])
                logger.info(f"OCR detected {len(ocr_data.get('mailbox_text', []))} mailbox texts")
            
            # Normalize building markers
            if 'building_markers' in ocr_data and isinstance(ocr_data['building_markers'], list):
                for item in ocr_data['building_markers']:
                    if isinstance(item, dict) and 'text' in item:
                        item['normalized'] = normalize_ocr_text(item['text'])
                logger.info(f"OCR detected {len(ocr_data.get('building_markers', []))} building markers")
            
            # Log confidence scores
            if 'confidence_scores' in ocr_data:
                scores = ocr_data['confidence_scores']
                logger.info(f"OCR confidence - House numbers: {scores.get('house_numbers', 0)}%, "
                          f"Street signs: {scores.get('street_signs', 0)}%, "
                          f"Mailbox: {scores.get('mailbox_text', 0)}%, "
                          f"Building markers: {scores.get('building_markers', 0)}%")
        
            # Extract address components from all detected text
            all_detected_text = []
            for field in ['house_numbers', 'street_signs', 'mailbox_text', 'building_markers']:
                if field in ocr_data and isinstance(ocr_data[field], list):
                    for item in ocr_data[field]:
                        if isinstance(item, dict) and 'text' in item:
                            all_detected_text.append(item['text'])
            
            if all_detected_text:
                structured_components = extract_address_components(all_detected_text)
                analysis['structured_address_components'] = structured_components
                logger.info(f"Structured address extraction: {structured_components}")
        
        # Log property features if detected
        if 'property_features' in analysis:
            features = analysis['property_features']
            logger.info(f"Detected property type: {features.get('property_type')}")
            if features.get('bedrooms'):
                logger.info(f"Estimated bedrooms: {features.get('bedrooms')}")
            if features.get('bathrooms'):
                logger.info(f"Estimated bathrooms: {features.get('bathrooms')}")
            if features.get('special_features'):
                logger.info(f"Special features: {', '.join(features.get('special_features', []))}")
        
        # Cache the successful result
        _cache_result(cache_key, analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in photo analysis: {str(e)}")
        return {
            "visual_address_clues": [],
            "detected_text": [],
            "landmarks": [],  
            "building_features": [],
            "ocr_address_data": {
                "house_numbers": [],
                "street_signs": [],
                "mailbox_text": [],
                "building_markers": [],
                "confidence_scores": {
                    "house_numbers": 0,
                    "street_signs": 0,
                    "mailbox_text": 0,
                    "building_markers": 0
                }
            },
            "property_features": {
                "bedrooms": None,
                "bathrooms": None,
                "property_type": None,
                "style": None,
                "special_features": [],
                "floor_type": None,
                "kitchen_features": [],
                "view": None,
                "stories": None,
                "exterior": None
            },
            "estimated_location": None,
            "confidence_score": 0,
            "error": str(e)
        }

def cross_reference_visual_with_maps(visual_clues: Dict[str, Any], lat: float, lng: float) -> Dict[str, Any]:
    """
    Cross-reference visual clues with Google Maps data for the location.
    
    Args:
        visual_clues: Visual analysis results
        lat: Latitude
        lng: Longitude
        
    Returns:
        Verification results
    """
    try:
        # Use Google Maps API if available
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            return {"verified": False, "reason": "No Google Maps API key"}
        
        import googlemaps
        gmaps = googlemaps.Client(key=api_key)
        
        # Get nearby places
        nearby_places = gmaps.places_nearby(  # type: ignore
            location=(lat, lng),
            radius=200,  # 200 meter radius
            language="en"
        )
        
        nearby_names = []
        if nearby_places.get("results"):
            nearby_names = [place.get("name", "") for place in nearby_places["results"][:10]]
        
        # Check if any detected text matches nearby places
        detected_text = visual_clues.get("detected_text", [])
        landmarks = visual_clues.get("landmarks", [])
        
        matches = []
        for text in detected_text + landmarks:
            for place in nearby_names:
                if place.lower() in text.lower() or text.lower() in place.lower():
                    matches.append(f"{text} matches {place}")
        
        return {
            "verified": len(matches) > 0,
            "matches": matches,
            "nearby_places": nearby_names[:5]
        }
        
    except Exception as e:
        logger.error(f"Error in cross-referencing: {str(e)}")
        return {"verified": False, "error": str(e)}

def extract_address_from_visual_context(photos: List[str], known_coords: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Main function to extract and verify address using visual analysis with OCR.
    Now enhanced with Tesseract OCR for better house number detection.
    
    Args:
        photos: List of photo URLs
        known_coords: Optional (lat, lng) tuple for verification
        
    Returns:
        Complete visual address extraction results including OCR data and property features
    """
    # Generate cache key for this request
    coords_str = f"{known_coords[0]},{known_coords[1]}" if known_coords else "no-coords"
    photos_str = '|'.join(sorted(photos[:5]))  # Use first 5 photos for cache key
    cache_key = hashlib.sha256(f"extract_context:{photos_str}:{coords_str}".encode()).hexdigest()
    
    # Check cache first for performance optimization
    if cache_key in _vision_cache:
        cached_data, timestamp = _vision_cache[cache_key]
        if time.time() - timestamp < VISION_CACHE_TTL:
            logger.info("Returning cached vision analysis results (cache hit)")
            return cached_data
        else:
            # Remove expired cache entry
            del _vision_cache[cache_key]
            logger.info("Cache entry expired, performing fresh analysis")
    
    result = {
        "visual_analysis": {},
        "ocr_summary": {},
        "tesseract_ocr": {},
        "cross_reference": {},
        "final_address_confidence": 0,
        "suggested_address": None,
        "property_match_confidence": 0,
        "ocr_confidence": 0
    }
    
    # Run all OCR systems in parallel for speed
    logger.info("Starting parallel OCR processing with all three systems...")
    start_time = time.time()
    
    # Initialize results containers with default structure
    visual_analysis = {
        "visual_address_clues": [],
        "detected_text": [],
        "landmarks": [],
        "building_features": [],
        "ocr_address_data": {
            "house_numbers": [],
            "street_signs": [],
            "mailbox_text": [],
            "building_markers": [],
            "confidence_scores": {
                "house_numbers": 0,
                "street_signs": 0,
                "mailbox_text": 0,
                "building_markers": 0
            }
        },
        "property_features": {
            "bedrooms": None,
            "bathrooms": None,
            "property_type": None,
            "style": None,
            "special_features": [],
            "floor_type": None,
            "kitchen_features": [],
            "view": None,
            "stories": None,
            "exterior": None
        },
        "estimated_location": None,
        "confidence_score": 0
    }
    tesseract_results = {}
    google_vision_results = {}
    
    # Define worker functions for parallel execution with timeout protection
    def run_gpt4_vision():
        try:
            start = time.time()
            logger.info("Starting GPT-4 Vision analysis...")
            
            # Limit photos to first 3 for faster processing
            limited_photos = photos[:3] if len(photos) > 3 else photos
            result = analyze_property_photos(limited_photos)
            
            duration = time.time() - start
            logger.info(f"GPT-4 Vision completed in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in GPT-4 Vision: {str(e)}")
            return {"error": str(e)}
    
    def run_tesseract():
        try:
            start = time.time()
            from tesseract_ocr import enhance_with_tesseract
            logger.info("Starting Tesseract OCR...")
            
            # Use only 3 photos for speed and timeout safety
            limited_photos = photos[:3] if len(photos) > 3 else photos
            result = enhance_with_tesseract(limited_photos, focus_on_photo_11=False)
            
            duration = time.time() - start
            logger.info(f"Tesseract OCR completed in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {str(e)}")
            return {"error": str(e)}
    
    def run_google_vision():
        try:
            start = time.time()
            from google_vision_ocr import analyze_with_google_vision, is_google_vision_available
            if is_google_vision_available():
                logger.info("Starting Google Cloud Vision OCR...")
                
                # Use only 3 photos for speed and timeout safety
                limited_photos = photos[:3] if len(photos) > 3 else photos
                result = analyze_with_google_vision(limited_photos)
                
                duration = time.time() - start
                logger.info(f"Google Cloud Vision completed in {duration:.2f}s")
                return result
            else:
                logger.info("Google Cloud Vision not available")
                return {"status": "not_configured"}
        except Exception as e:
            logger.error(f"Error in Google Vision OCR: {str(e)}")
            return {"error": str(e)}
    
    # Execute all OCR systems in parallel with proper timeout handling
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        # Track submission times for timeout monitoring
        submission_time = time.time()
        future_to_ocr = {
            executor.submit(run_gpt4_vision): 'gpt4',
            executor.submit(run_tesseract): 'tesseract',
            executor.submit(run_google_vision): 'google_vision'
        }
        
        # Log that we're starting the parallel execution
        logger.info(f"Submitted {len(future_to_ocr)} OCR tasks at {submission_time}")
        
        # Use FIRST_COMPLETED strategy with a 10-second timeout to stay within worker limits
        # This processes results as they become available instead of blocking
        import concurrent.futures
        remaining_futures = set(future_to_ocr.keys())
        timeout_deadline = time.time() + 10  # 10 seconds total timeout
        
        while remaining_futures and time.time() < timeout_deadline:
            # Calculate remaining timeout
            remaining_timeout = max(0.1, timeout_deadline - time.time())
            
            # Wait for the next future(s) to complete with the remaining timeout
            done, pending = concurrent.futures.wait(
                remaining_futures,
                timeout=min(remaining_timeout, 2),  # Check every 2 seconds max
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed futures immediately
            for future in done:
                ocr_type = future_to_ocr[future]
                processing_time = time.time() - submission_time
                try:
                    # Don't use timeout here since the future is already done
                    result_data = future.result(timeout=0.1)
                    if ocr_type == 'gpt4':
                        visual_analysis = result_data
                    elif ocr_type == 'tesseract':
                        tesseract_results = result_data
                    elif ocr_type == 'google_vision':
                        google_vision_results = result_data
                    logger.info(f"Completed {ocr_type} OCR in {processing_time:.2f}s")
                except Exception as e:
                    logger.error(f"Failed to get result for {ocr_type} after {processing_time:.2f}s: {str(e)}")
                
                # Remove from remaining futures
                remaining_futures.discard(future)
            
            # Update remaining futures to be pending ones
            remaining_futures = pending
        
        # Cancel any remaining futures that haven't completed within timeout
        if remaining_futures:
            logger.warning(f"Timeout reached after 10 seconds, cancelling {len(remaining_futures)} pending OCR tasks")
            for future in remaining_futures:
                ocr_type = future_to_ocr[future]
                logger.warning(f"Cancelling {ocr_type} OCR due to timeout (exceeded 10s limit)")
                # Cancel the future - this will attempt to stop the thread if it hasn't started
                cancelled = future.cancel()
                if not cancelled:
                    # If we couldn't cancel (task already running), we still need to handle it
                    logger.warning(f"Could not cancel {ocr_type} (already running), will wait in shutdown")
                    
    except Exception as e:
        logger.error(f"Error in parallel OCR execution: {str(e)}")
    finally:
        # Shutdown the executor with a short wait to avoid blocking
        # If tasks are still running, we'll forcefully cancel them
        logger.info("Shutting down executor...")
        try:
            # Try to shutdown gracefully with a 1-second wait
            executor.shutdown(wait=False, cancel_futures=True)
            # Give threads a moment to terminate
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error during executor shutdown: {e}")
        finally:
            # Executor is shutting down, no need to manually clear threads
            logger.info("Executor shutdown complete")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Parallel OCR processing completed in {elapsed_time:.2f} seconds")
    
    # Store results
    result["visual_analysis"] = visual_analysis
    result["tesseract_ocr"] = tesseract_results
    result["google_vision_ocr"] = google_vision_results
    
    # Merge Tesseract results into visual analysis
    if tesseract_results and not tesseract_results.get('error'):
        best_house_number = tesseract_results.get('best_house_number')
        if best_house_number and isinstance(best_house_number, dict):
            logger.info(f"Tesseract detected house number: {best_house_number.get('text')} with confidence {best_house_number.get('confidence')}")
            
            # Ensure OCR data structure exists and is a dict
            if not isinstance(visual_analysis, dict):
                visual_analysis = {}
            
            # Type assertion for visual_analysis
            va_dict = visual_analysis  # This is definitely a dict now
            ocr_data = va_dict.get("ocr_address_data")
            if not isinstance(ocr_data, dict):
                ocr_data = {
                    "house_numbers": [],
                    "street_signs": [],
                    "mailbox_text": [],
                    "building_markers": [],
                    "confidence_scores": {}
                }
                va_dict["ocr_address_data"] = ocr_data  # type: ignore
            
            # Add Tesseract-detected house numbers
            house_numbers_list = ocr_data.get("house_numbers", [])
            if not isinstance(house_numbers_list, list):
                house_numbers_list = []
                ocr_data["house_numbers"] = house_numbers_list
                
            for house_num in tesseract_results.get('all_house_numbers', []):
                house_numbers_list.append({
                    "text": house_num,
                    "confidence": best_house_number.get('confidence', 80),
                    "location": f"Photo {best_house_number.get('photo_index', 'unknown')}",
                    "source": "Tesseract OCR"
                })
            
            # Update confidence scores
            confidence_scores = ocr_data.get("confidence_scores", {})
            if not isinstance(confidence_scores, dict):
                confidence_scores = {}
                ocr_data["confidence_scores"] = confidence_scores
            confidence_scores["house_numbers"] = best_house_number.get('confidence', 0)
    
    # Merge Google Vision results into visual analysis
    if google_vision_results and not google_vision_results.get('error') and google_vision_results.get('status') != 'not_configured':
        # Ensure OCR data structure exists and is a dict
        if not isinstance(visual_analysis, dict):
            visual_analysis = {}
        
        # Type assertion for visual_analysis
        va_dict = visual_analysis  # This is definitely a dict now
        ocr_data = va_dict.get("ocr_address_data")
        if not isinstance(ocr_data, dict):
            ocr_data = {
                "house_numbers": [],
                "street_signs": [],
                "mailbox_text": [],
                "building_markers": [],
                "confidence_scores": {}
            }
            va_dict["ocr_address_data"] = ocr_data  # type: ignore
        
        # Process Google Vision house numbers
        best_gv_number = google_vision_results.get('best_house_number')
        if best_gv_number and isinstance(best_gv_number, dict):
            logger.info(f"Google Vision detected house number: {best_gv_number.get('text')} with confidence {best_gv_number.get('confidence')}")
            
            # Get house numbers list
            house_numbers_list = ocr_data.get("house_numbers", [])
            if not isinstance(house_numbers_list, list):
                house_numbers_list = []
                ocr_data["house_numbers"] = house_numbers_list
            
            # Add Google Vision detected house numbers
            for house_num in google_vision_results.get('all_house_numbers', []):
                # Check if this number is already in the list
                existing = any(h.get('text') == house_num for h in house_numbers_list if isinstance(h, dict))
                if not existing:
                    house_numbers_list.append({
                        "text": house_num,
                        "confidence": best_gv_number.get('confidence', 85),
                        "location": f"Photo {best_gv_number.get('photo_index', 'unknown')}",
                        "source": "Google Cloud Vision"
                    })
            
            # Update confidence scores - take the maximum between existing and Google Vision
            confidence_scores = ocr_data.get("confidence_scores", {})
            if not isinstance(confidence_scores, dict):
                confidence_scores = {}
                ocr_data["confidence_scores"] = confidence_scores
                
            current_confidence = confidence_scores.get("house_numbers", 0)
            gv_confidence = best_gv_number.get('confidence', 0)
            
            # Use the higher confidence score (best of both systems)
            if gv_confidence > current_confidence:
                confidence_scores["house_numbers"] = gv_confidence
                logger.info(f"Updated OCR confidence to Google Vision score: {gv_confidence}")
            
            # If Google Vision found a house number with high confidence, prioritize it
            if gv_confidence > 80:
                # Move Google Vision results to the front
                gv_houses = [h for h in house_numbers_list if isinstance(h, dict) and h.get('source') == 'Google Cloud Vision']
                other_houses = [h for h in house_numbers_list if isinstance(h, dict) and h.get('source') != 'Google Cloud Vision']
                ocr_data["house_numbers"] = gv_houses + other_houses
        
        # Also extract street names from Google Vision full text
        if google_vision_results.get('photo_results'):
            for photo_result in google_vision_results.get('photo_results', []):
                if isinstance(photo_result, dict) and photo_result.get('full_text'):
                    # Extract potential street names from full text
                    street_patterns = [
                        r'\b(\w+)\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir)\b',
                        r'\b(\w+\s+\w+)\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir)\b'
                    ]
                    
                    # Get street signs list
                    street_signs_list = ocr_data.get("street_signs", [])
                    if not isinstance(street_signs_list, list):
                        street_signs_list = []
                        ocr_data["street_signs"] = street_signs_list
                    
                    for pattern in street_patterns:
                        matches = re.finditer(pattern, photo_result['full_text'], re.IGNORECASE)
                        for match in matches:
                            street_name = match.group(0)
                            # Add to street signs if not already present
                            existing = any(s.get('text', '').lower() == street_name.lower() 
                                         for s in street_signs_list if isinstance(s, dict))
                            if not existing:
                                street_signs_list.append({
                                    "text": street_name,
                                    "confidence": 75,
                                    "location": f"Photo {photo_result.get('photo_index', 'unknown')}",
                                    "source": "Google Cloud Vision"
                                })
                                logger.info(f"Google Vision detected street: {street_name}")
    
    # Process OCR data if available
    ocr_data = visual_analysis.get("ocr_address_data")
    if isinstance(ocr_data, dict):
        # Create OCR summary
        ocr_summary = {
            "found_house_numbers": [],
            "found_street_names": [],
            "found_mailbox_info": [],
            "found_building_names": [],
            "overall_ocr_confidence": 0
        }
        
        # Extract house numbers
        house_numbers = ocr_data.get("house_numbers", [])
        if isinstance(house_numbers, list):
            for item in house_numbers:
                if isinstance(item, dict):
                    ocr_summary["found_house_numbers"].append({
                        "text": item.get("normalized", item.get("text", "")),
                        "confidence": item.get("confidence", 0)
                    })
        
        # Extract street signs
        street_signs = ocr_data.get("street_signs", [])
        if isinstance(street_signs, list):
            for item in street_signs:
                if isinstance(item, dict):
                    ocr_summary["found_street_names"].append({
                        "text": item.get("normalized", item.get("text", "")),
                        "confidence": item.get("confidence", 0)
                    })
        
        # Extract mailbox text
        mailbox_text = ocr_data.get("mailbox_text", [])
        if isinstance(mailbox_text, list):
            for item in mailbox_text:
                if isinstance(item, dict):
                    ocr_summary["found_mailbox_info"].append({
                        "text": item.get("normalized", item.get("text", "")),
                        "confidence": item.get("confidence", 0)
                    })
        
        # Extract building markers
        building_markers = ocr_data.get("building_markers", [])
        if isinstance(building_markers, list):
            for item in building_markers:
                if isinstance(item, dict):
                    ocr_summary["found_building_names"].append({
                        "text": item.get("normalized", item.get("text", "")),
                        "confidence": item.get("confidence", 0)
                    })
        
        # Calculate overall OCR confidence
        confidence_scores = ocr_data.get("confidence_scores", {})
        if isinstance(confidence_scores, dict):
            ocr_confidences = [
                confidence_scores.get("house_numbers", 0),
                confidence_scores.get("street_signs", 0),
                confidence_scores.get("mailbox_text", 0),
                confidence_scores.get("building_markers", 0)
            ]
            # Weight house numbers and street signs more heavily
            weighted_confidence = (
                ocr_confidences[0] * 0.35 +  # house numbers - 35%
                ocr_confidences[1] * 0.35 +  # street signs - 35%
                ocr_confidences[2] * 0.15 +  # mailbox - 15%
                ocr_confidences[3] * 0.15     # building markers - 15%
            )
            ocr_summary["overall_ocr_confidence"] = int(weighted_confidence)
            result["ocr_confidence"] = int(weighted_confidence)
        
        result["ocr_summary"] = ocr_summary
        
        # Try to construct address from OCR data
        if ocr_summary["found_house_numbers"] or ocr_summary["found_street_names"]:
            address_parts = []
            
            # Add highest confidence house number
            if ocr_summary["found_house_numbers"]:
                best_number = max(ocr_summary["found_house_numbers"], 
                                key=lambda x: x.get("confidence", 0))
                if best_number["confidence"] > 50:
                    address_parts.append(best_number["text"])
            
            # Add highest confidence street name
            if ocr_summary["found_street_names"]:
                best_street = max(ocr_summary["found_street_names"], 
                                key=lambda x: x.get("confidence", 0))
                if best_street["confidence"] > 50:
                    address_parts.append(best_street["text"])
            
            # Construct suggested address from OCR
            if address_parts:
                ocr_suggested_address = " ".join(address_parts)
                if not result.get("suggested_address"):
                    result["suggested_address"] = ocr_suggested_address
                else:
                    # Combine OCR address with other visual analysis
                    result["suggested_address_ocr"] = ocr_suggested_address
    
    # Calculate property feature confidence
    property_features = visual_analysis.get("property_features")
    if isinstance(property_features, dict):
        feature_count = 0
        total_features = 10  # Total possible feature categories
        
        # Count non-null/non-empty features
        if property_features.get("bedrooms") is not None:
            feature_count += 1
        if property_features.get("bathrooms") is not None:
            feature_count += 1
        if property_features.get("property_type"):
            feature_count += 1
        if property_features.get("style"):
            feature_count += 1
        if property_features.get("special_features"):
            feature_count += 1
        if property_features.get("floor_type"):
            feature_count += 1
        if property_features.get("kitchen_features"):
            feature_count += 1
        if property_features.get("view"):
            feature_count += 1
        if property_features.get("stories") is not None:
            feature_count += 1
        if property_features.get("exterior"):
            feature_count += 1
            
        # Property match confidence based on detected features
        result["property_match_confidence"] = int((feature_count / total_features) * 100)
    
    # Cross-reference if coordinates available (disabled to prevent timeout)
    if False and known_coords and len(known_coords) == 2:
        cross_ref = cross_reference_visual_with_maps(visual_analysis, known_coords[0], known_coords[1])
        result["cross_reference"] = cross_ref
        
        # Adjust confidence based on location, OCR, and property features
        location_confidence = visual_analysis.get("confidence_score", 0)
        ocr_confidence = result.get("ocr_confidence", 0)
        
        if cross_ref.get("verified"):
            location_confidence = min(100, location_confidence + 30)
        
        # Combined confidence (40% location, 35% OCR, 25% property features)
        result["final_address_confidence"] = int(
            (location_confidence * 0.4) + 
            (ocr_confidence * 0.35) +
            (result.get("property_match_confidence", 0) * 0.25)
        )
    else:
        # When no coordinates (50% location, 35% OCR, 15% property features)
        confidence_score = 0
        if isinstance(visual_analysis, dict):
            raw_score = visual_analysis.get("confidence_score", 0)
            # Ensure it's a number, not a string
            if isinstance(raw_score, (int, float)):
                confidence_score = raw_score
            elif isinstance(raw_score, str):
                try:
                    confidence_score = float(raw_score)
                except (ValueError, TypeError):
                    confidence_score = 0
        
        result["final_address_confidence"] = int(
            (confidence_score * 0.5) + 
            (result.get("ocr_confidence", 0) * 0.35) +
            (result.get("property_match_confidence", 0) * 0.15)
        )
    
    # Set suggested address if found
    if isinstance(visual_analysis, dict) and visual_analysis.get("estimated_location") and not result.get("suggested_address"):
        result["suggested_address"] = visual_analysis.get("estimated_location")
    
    # Ensure landmarks and building_features are included in the final result
    if isinstance(visual_analysis, dict):
        # Include landmarks array (ensure it's always a list)
        result["landmarks"] = visual_analysis.get("landmarks", [])
        if not isinstance(result["landmarks"], list):
            result["landmarks"] = []
            
        # Include building_features array (ensure it's always a list)
        result["building_features"] = visual_analysis.get("building_features", [])
        if not isinstance(result["building_features"], list):
            result["building_features"] = []
            
        # Include environmental context if available
        if "environmental_context" in visual_analysis:
            result["environmental_context"] = visual_analysis.get("environmental_context", {})
            
        # Log summary of landmarks and features
        landmark_count = len(result["landmarks"])
        feature_count = len(result["building_features"])
        if landmark_count > 0 or feature_count > 0:
            logger.info(f"Final result includes {landmark_count} landmarks and {feature_count} building features")
    else:
        # Ensure empty arrays if visual_analysis is not available
        result["landmarks"] = []
        result["building_features"] = []
    
    # Cache the results before returning
    _vision_cache[cache_key] = (result, time.time())
    # Clean up old cache entries (keep max 100 entries)
    if len(_vision_cache) > 100:
        # Remove oldest entries
        sorted_entries = sorted(_vision_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_entries[:20]:  # Remove 20 oldest
            del _vision_cache[old_key]
    logger.info(f"Cached vision analysis results (cache key: {cache_key[:8]}...)")
    
    return result