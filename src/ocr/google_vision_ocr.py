"""
Google Cloud Vision OCR Module for Enhanced House Number Detection
This module uses Google's state-of-the-art Cloud Vision API to detect house numbers
with superior accuracy compared to traditional OCR methods.
"""

import os
import json
import logging
import re
import io
from typing import List, Dict, Optional, Any, Tuple
from google.cloud import vision
from google.oauth2 import service_account
import requests
from PIL import Image
import numpy as np
from src.utils.resilience import resilient_google_vision_call, resilience_manager

logger = logging.getLogger(__name__)

class GoogleVisionOCR:
    """Google Cloud Vision OCR for detecting house numbers in property photos"""
    
    def __init__(self):
        """Initialize Google Cloud Vision client with service account credentials"""
        try:
            # Get credentials from environment variable
            creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if not creds_json:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")
                self.client = None
                return
                
            # Parse JSON credentials
            creds_dict = json.loads(creds_json)
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/cloud-vision']
            )
            
            # Initialize Vision client with credentials
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("Google Cloud Vision client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision: {str(e)}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Google Cloud Vision is available"""
        return self.client is not None
    
    def download_image_bytes(self, url: str) -> Optional[bytes]:
        """Download image from URL and return bytes"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
        return None
    
    @resilient_google_vision_call
    def detect_text(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Enhanced text detection with improved confidence scoring and position analysis
        
        Returns:
            Dictionary containing detected text, house numbers, and confidence scores
        """
        if not self.client:
            logger.warning("Google Cloud Vision not available, using Tesseract fallback")
            from src.ocr.tesseract_ocr import tesseract_ocr
            return tesseract_ocr.detect_text(image_bytes)
        
        try:
            # Create Vision image object
            image = vision.Image(content=image_bytes)
            
            # Perform text detection with language hints for better accuracy
            image_context = vision.ImageContext(
                language_hints=['en'],  # English text
                text_detection_params=vision.TextDetectionParams(
                    enable_text_detection_confidence_score=True
                )
            )
            
            # Perform text detection
            response = self.client.text_detection(image=image, image_context=image_context)
            texts = response.text_annotations
            
            # Process detected text
            all_text_items = []
            house_numbers = set()
            house_number_candidates = []
            street_names = []
            
            # First annotation contains the full text
            if texts:
                full_text = texts[0].description
                logger.info(f"Full text detected: {full_text[:100]}...")
                
                # Extract potential street names from full text
                street_patterns = [
                    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b',
                    r'\b\d+\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Street|St|Avenue|Ave|Road|Rd)\b'
                ]
                for pattern in street_patterns:
                    matches = re.finditer(pattern, full_text, re.IGNORECASE)
                    for match in matches:
                        street_names.append(match.group(0))
                
                # Process individual text annotations
                for i, text in enumerate(texts[1:], 1):  # Skip first (full text)
                    text_str = text.description.strip()
                    
                    # Enhanced confidence calculation
                    vertices = text.bounding_poly.vertices
                    area = self._calculate_polygon_area(vertices)
                    
                    # Calculate position-based confidence boost
                    position_boost = 0
                    if vertices:
                        # Text in upper portion of image gets boost (house numbers often at top)
                        avg_y = sum(v.y for v in vertices) / len(vertices)
                        if avg_y < 500:  # Upper portion
                            position_boost = 10
                        
                        # Text in central horizontal region gets boost
                        avg_x = sum(v.x for v in vertices) / len(vertices)
                        if 200 < avg_x < 800:  # Central region
                            position_boost += 5
                    
                    # Calculate sharpness confidence (larger, clearer text = higher confidence)
                    base_confidence = min(100, max(30, int(area / 100)))
                    
                    # Apply position boost
                    confidence = min(100, base_confidence + position_boost)
                    
                    # Additional confidence boost for text that looks like addresses
                    if self.is_house_number(text_str):
                        confidence = min(100, confidence + 10)
                    elif self._is_street_component(text_str):
                        confidence = min(100, confidence + 5)
                    
                    text_item = {
                        'text': text_str,
                        'confidence': confidence,
                        'base_confidence': base_confidence,
                        'position_boost': position_boost,
                        'bounds': {
                            'vertices': [(v.x, v.y) for v in vertices]
                        },
                        'area': area
                    }
                    all_text_items.append(text_item)
                    
                    # Check if it's a house number
                    if self.is_house_number(text_str):
                        house_numbers.add(text_str)
                        house_number_candidates.append({
                            'text': text_str,
                            'confidence': confidence,
                            'source': 'google_vision',
                            'bounds': text_item['bounds'],
                            'area': area
                        })
                        logger.info(f"Found house number: {text_str} (confidence: {confidence}, position_boost: {position_boost})")
            
            # Sort candidates by confidence
            house_number_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Check for errors
            if response.error.message:
                logger.error(f"Google Vision API error: {response.error.message}")
                return {
                    'error': response.error.message,
                    'house_numbers_found': [],
                    'all_text': []
                }
            
            return {
                'house_numbers_found': list(house_numbers),
                'best_candidate': house_number_candidates[0] if house_number_candidates else None,
                'all_candidates': house_number_candidates[:5],
                'all_text': all_text_items[:30],  # Return top 30 text items
                'street_names': street_names,  # Added detected street names
                'full_text': texts[0].description if texts else "",
                'text_count': len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error in Google Vision text detection: {str(e)}")
            return {
                'error': str(e),
                'house_numbers_found': [],
                'all_text': []
            }
    
    def _calculate_polygon_area(self, vertices) -> float:
        """Calculate area of polygon from vertices (for confidence estimation)"""
        if not vertices or len(vertices) < 3:
            return 0
        
        # Shoelace formula for polygon area
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y
            area -= vertices[j].x * vertices[i].y
        return abs(area) / 2
    
    def is_house_number(self, text: str) -> bool:
        """
        Check if text looks like a house number
        More comprehensive than Tesseract's version
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Patterns for house numbers (expanded)
        patterns = [
            r'^\d{1,5}[A-Za-z]?$',  # 1-5 digits optionally followed by a letter
            r'^\d{1,4}-\d{1,2}$',    # Numbers with dash (e.g., 123-45)
            r'^\d{1,3}\s?[A-Za-z]$',  # Number with space and letter
            r'^\d{1,5}$',             # Just digits
            r'^#\d{1,5}$',            # With hash prefix
            r'^\d{1,3}[/-]\d{1,3}$',  # Fraction or range style
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                try:
                    # Extract numeric part
                    num_parts = re.findall(r'\d+', text)
                    if num_parts:
                        num_val = int(num_parts[0])
                        # House numbers typically between 1 and 99999
                        if 1 <= num_val <= 99999:
                            return True
                except:
                    pass
        
        return False
    
    def _is_street_component(self, text: str) -> bool:
        """
        Check if text looks like a street name component
        """
        if not text:
            return False
        
        text = text.strip().lower()
        
        # Common street suffixes
        street_suffixes = [
            'street', 'st', 'avenue', 'ave', 'road', 'rd',
            'boulevard', 'blvd', 'drive', 'dr', 'lane', 'ln',
            'way', 'court', 'ct', 'place', 'pl', 'circle', 'cir',
            'trail', 'trl', 'parkway', 'pkwy', 'highway', 'hwy',
            'terrace', 'ter', 'plaza', 'plz', 'square', 'sq'
        ]
        
        # Check if text is or contains a street suffix
        for suffix in street_suffixes:
            if text == suffix or text.endswith(' ' + suffix):
                return True
        
        # Check if text looks like a street name pattern
        street_patterns = [
            r'^[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd)',
            r'^\d+(?:st|nd|rd|th)\s+(?:Street|St|Avenue|Ave)',
            r'^(?:North|South|East|West|N|S|E|W)\s+[A-Z]'
        ]
        
        for pattern in street_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    @resilient_google_vision_call
    def detect_document_text(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Alternative: Use document text detection for better structure understanding
        This can be more accurate for certain types of text
        """
        if not self.client:
            logger.warning("Google Cloud Vision not available, using Tesseract fallback")
            from src.ocr.tesseract_ocr import tesseract_ocr
            return tesseract_ocr.detect_text(image_bytes)
        
        try:
            image = vision.Image(content=image_bytes)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                return {'error': response.error.message}
            
            # Process document structure
            document = response.full_text_annotation
            text_items = []
            
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        para_text = ""
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            para_text += word_text + " "
                        
                        para_text = para_text.strip()
                        if para_text:
                            text_items.append({
                                'text': para_text,
                                'confidence': paragraph.confidence * 100 if hasattr(paragraph, 'confidence') else 80
                            })
            
            return {
                'document_text': document.text if document else "",
                'structured_text': text_items
            }
            
        except Exception as e:
            logger.error(f"Error in document text detection: {str(e)}")
            return {'error': str(e)}
    
    def analyze_photos(self, photo_urls: List[str], max_photos: int = 15) -> Dict[str, Any]:
        """
        Analyze multiple photos for house numbers using Google Cloud Vision
        
        Args:
            photo_urls: List of photo URLs from Airbnb
            max_photos: Maximum number of photos to analyze (default 15)
        
        Returns:
            Dictionary with combined results from all photos
        """
        if not self.client:
            logger.warning("Google Cloud Vision not available, skipping analysis")
            return {
                'error': 'Google Cloud Vision not configured',
                'all_house_numbers': [],
                'photo_results': []
            }
        
        all_results = []
        combined_house_numbers = set()
        best_overall_candidate = None
        best_confidence = 0
        
        # Analyze first N photos
        photos_to_analyze = photo_urls[:min(max_photos, len(photo_urls))]
        
        for idx, url in enumerate(photos_to_analyze):
            logger.info(f"Analyzing photo {idx + 1}/{len(photos_to_analyze)} with Google Cloud Vision")
            
            # Download image
            image_bytes = self.download_image_bytes(url)
            if not image_bytes:
                continue
            
            # Detect text using regular text detection (faster and better for scene text)
            results = self.detect_text(image_bytes)
            results['photo_index'] = idx + 1
            results['url'] = url
            
            # Also try document text detection for first few photos (more thorough)
            if idx < 3:  # Only for first 3 photos to save API calls
                doc_results = self.detect_document_text(image_bytes)
                results['document_analysis'] = doc_results
            
            all_results.append(results)
            
            # Combine house numbers
            if 'house_numbers_found' in results:
                combined_house_numbers.update(results['house_numbers_found'])
            
            # Track best candidate
            if results.get('best_candidate') and results['best_candidate'].get('confidence', 0) > best_confidence:
                best_overall_candidate = results['best_candidate']
                best_overall_candidate['photo_index'] = idx + 1
                best_confidence = results['best_candidate']['confidence']
        
        # Create summary
        summary = {
            'all_house_numbers': list(combined_house_numbers),
            'best_house_number': best_overall_candidate,
            'photo_results': all_results,
            'summary': {
                'photos_analyzed': len(all_results),
                'house_numbers_found': len(combined_house_numbers),
                'best_confidence': best_confidence,
                'api_calls_made': len(all_results) + sum(1 for r in all_results if 'document_analysis' in r)
            }
        }
        
        logger.info(f"Google Vision OCR complete: Found {len(combined_house_numbers)} unique house numbers")
        if best_overall_candidate:
            logger.info(f"Best candidate: {best_overall_candidate['text']} with confidence {best_confidence}")
        
        return summary

# Create singleton instance
google_vision_ocr = GoogleVisionOCR()

def analyze_with_google_vision(photo_urls: List[str]) -> Dict[str, Any]:
    """
    Main entry point for Google Cloud Vision OCR analysis
    
    Args:
        photo_urls: List of photo URLs from Airbnb listing
    
    Returns:
        Dictionary with OCR results including house numbers found
    """
    return google_vision_ocr.analyze_photos(photo_urls)

def is_google_vision_available() -> bool:
    """Check if Google Cloud Vision is configured and available"""
    return google_vision_ocr.is_available()