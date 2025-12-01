"""
Tesseract OCR Module for Enhanced House Number Detection
This module uses Tesseract OCR with image preprocessing to detect house numbers
that GPT-4 Vision might miss, especially small or subtle numbers.
"""

import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image
import io
import requests
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class TesseractHouseNumberDetector:
    """Specialized OCR for detecting house numbers in property photos"""
    
    def __init__(self):
        """Initialize Tesseract with optimized settings for number detection"""
        # Configure Tesseract for better number detection
        self.custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        
    def download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to OpenCV format"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # Convert PIL to OpenCV format
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            return None
            
    def preprocess_for_house_numbers(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply comprehensive preprocessing techniques for optimal text detection"""
        processed_images = []
        
        # Convert to grayscale - essential first step
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray)
        
        # 1. Enhanced CLAHE with multiple settings for better contrast
        # Standard CLAHE
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img1 = clahe1.apply(gray)
        processed_images.append(clahe_img1)
        
        # Aggressive CLAHE for very low contrast
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        clahe_img2 = clahe2.apply(gray)
        processed_images.append(clahe_img2)
        
        # 2. Multiple adaptive thresholding variations for different lighting conditions
        # Standard adaptive threshold
        adaptive_thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive_thresh1)
        
        # Mean adaptive threshold (better for uniform lighting)
        adaptive_thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        processed_images.append(adaptive_thresh2)
        
        # Inverted adaptive threshold (for dark text on light background)
        adaptive_thresh_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        processed_images.append(adaptive_thresh_inv)
        
        # 3. Image sharpening using unsharp masking
        # Create a Gaussian blurred version
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        # Calculate the unsharp mask
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        processed_images.append(sharpened)
        
        # Apply sharpening kernel for edge enhancement
        sharpening_kernel = np.array([[-1,-1,-1],
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharpened2 = cv2.filter2D(gray, -1, sharpening_kernel)
        processed_images.append(sharpened2)
        
        # 4. Advanced noise reduction while preserving edges
        # Bilateral filter - preserves edges while reducing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(denoised)
        
        # Non-local means denoising (best edge preservation)
        denoised_nlm = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed_images.append(denoised_nlm)
        
        # 5. Morphological operations to improve text clarity
        # Opening operation to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        processed_images.append(morph_open)
        
        # 6. Contrast stretching
        # Normalize the image to full 0-255 range
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        processed_images.append(normalized)
        
        # 7. Gamma correction for different exposure levels
        # Gamma = 0.5 (brighten dark images)
        gamma = 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected1 = cv2.LUT(gray, table)
        processed_images.append(gamma_corrected1)
        
        # Gamma = 2.0 (darken bright images)
        gamma = 2.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected2 = cv2.LUT(gray, table)
        processed_images.append(gamma_corrected2)
        
        return processed_images  # Comprehensive preprocessing methods
    
    def extract_regions_of_interest(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract ONLY the most likely regions for house numbers (optimized for speed)"""
        height, width = image.shape[:2]
        regions = []
        
        # Full image first (for general detection)
        regions.append(image)
        
        # Upper half (house numbers usually in upper portion)
        upper_half = image[0:int(height*0.5), :]
        regions.append(upper_half)
        
        # Middle-right region (common for house numbers near doors)
        middle_right = image[int(height*0.3):int(height*0.7), int(width*0.5):]
        regions.append(middle_right)
        
        return regions  # Only 3 regions instead of 6
    
    def detect_house_numbers(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect house numbers using multiple preprocessing techniques"""
        found_numbers = set()
        all_text_found = []
        confidence_scores = []
        
        # Try different preprocessing techniques
        preprocessed_images = self.preprocess_for_house_numbers(image)
        
        for idx, processed_img in enumerate(preprocessed_images):
            try:
                # Get regions of interest
                regions = self.extract_regions_of_interest(processed_img)
                
                for region_idx, region in enumerate(regions):
                    # Skip if region is too small
                    if region.shape[0] < 20 or region.shape[1] < 20:
                        continue
                        
                    # Run Tesseract with ONLY the best PSM mode for house numbers
                    for psm_mode in [8, 11]:  # Only 2 modes: single word and sparse text
                        config = f'--oem 3 --psm {psm_mode}'
                        
                        try:
                            # Get detailed OCR data
                            data = pytesseract.image_to_data(region, config=config, output_type=pytesseract.Output.DICT)
                            
                            # Extract text with confidence
                            for i in range(len(data['text'])):
                                text = str(data['text'][i]).strip()
                                conf = int(data['conf'][i])
                                
                                if text and conf > 30:  # Lower threshold to catch more
                                    all_text_found.append({
                                        'text': text,
                                        'confidence': conf,
                                        'preprocessing': idx,
                                        'region': region_idx,
                                        'psm': psm_mode
                                    })
                                    
                                    # Check if it looks like a house number
                                    if self.is_house_number(text):
                                        found_numbers.add(text)
                                        confidence_scores.append(conf)
                                        logger.info(f"Found potential house number: {text} (confidence: {conf})")
                                        
                        except Exception as e:
                            continue
                            
            except Exception as e:
                logger.error(f"Error in OCR preprocessing {idx}: {str(e)}")
                continue
        
        # Find the most likely house number
        house_number_candidates = []
        for text_item in all_text_found:
            if self.is_house_number(text_item['text']):
                house_number_candidates.append(text_item)
        
        # Sort by confidence
        house_number_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'house_numbers_found': list(found_numbers),
            'best_candidate': house_number_candidates[0] if house_number_candidates else None,
            'all_candidates': house_number_candidates[:5],  # Top 5 candidates
            'all_text': all_text_found[:20],  # First 20 text items found
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0
        }
    
    def is_house_number(self, text: str) -> bool:
        """Check if text looks like a house number"""
        if not text:
            return False
            
        # Clean the text
        text = text.strip()
        
        # Patterns for house numbers
        patterns = [
            r'^\d{1,5}[A-Z]?$',  # 1-5 digits optionally followed by a letter
            r'^\d{1,4}-?\d{0,2}$',  # Numbers with optional dash
            r'^\d{1,3}\s?[A-Z]$',  # Number with space and letter
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                # Additional validation: should be a reasonable house number
                try:
                    # Extract just the numeric part
                    num_part = re.findall(r'\d+', text)[0]
                    num_val = int(num_part)
                    # House numbers are typically between 1 and 99999
                    if 1 <= num_val <= 99999:
                        return True
                except:
                    pass
        
        return False
    
    def analyze_photos(self, photo_urls: List[str], focus_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Analyze multiple photos for house numbers"""
        all_results = []
        combined_house_numbers = set()
        best_overall_candidate = None
        best_confidence = 0
        
        # If specific indices are provided, focus on those
        if focus_indices:
            photos_to_analyze = [(i, photo_urls[i]) for i in focus_indices if i < len(photo_urls)]
        else:
            photos_to_analyze = list(enumerate(photo_urls))
        
        for idx, url in photos_to_analyze:
            logger.info(f"Analyzing photo {idx + 1}/{len(photo_urls)} with Tesseract OCR")
            
            # Download image
            image = self.download_image(url)
            if image is None:
                continue
            
            # Detect house numbers
            results = self.detect_house_numbers(image)
            results['photo_index'] = idx + 1
            results['url'] = url
            
            all_results.append(results)
            combined_house_numbers.update(results['house_numbers_found'])
            
            # Track best candidate
            if results['best_candidate'] and results['best_candidate']['confidence'] > best_confidence:
                best_overall_candidate = results['best_candidate']
                best_overall_candidate['photo_index'] = idx + 1
                best_confidence = results['best_candidate']['confidence']
        
        return {
            'all_house_numbers': list(combined_house_numbers),
            'best_house_number': best_overall_candidate,
            'photo_results': all_results,
            'summary': {
                'photos_analyzed': len(all_results),
                'house_numbers_found': len(combined_house_numbers),
                'best_confidence': best_confidence
            }
        }

# Create singleton instance
tesseract_detector = TesseractHouseNumberDetector()

def enhance_with_tesseract(photo_urls: List[str], focus_on_photo_11: bool = False) -> Dict[str, Any]:
    """
    Enhanced OCR using Tesseract to find house numbers in exterior photos
    
    Args:
        photo_urls: List of photo URLs from Airbnb
        focus_on_photo_11: Legacy parameter (kept for compatibility)
    
    Returns:
        Dictionary with Tesseract OCR results
    """
    # Analyze first 15 photos - matching GPT-4 Vision coverage for comprehensive detection
    # This ensures we don't miss house numbers that appear later in the listing
    # Performance: 15 photos * 3 regions * 3 preprocessings * 2 PSM = 270 attempts
    focus_indices = list(range(min(15, len(photo_urls))))
    
    logger.info(f"Tesseract analyzing {len(focus_indices)} photos for house numbers (first {len(focus_indices)} photos)")
    return tesseract_detector.analyze_photos(photo_urls, focus_indices)