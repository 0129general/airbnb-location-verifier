"""
Street View Matcher Module
Compares Airbnb listing photos with Google Street View images to verify addresses

CURRENT STATUS: Disabled due to 30-second timeout constraints
FUTURE: Enable with async processing or background tasks

This module can significantly improve address accuracy by:
- Comparing Airbnb photos with Street View images at candidate addresses
- Identifying the correct house when multiple similar addresses exist
- Providing visual confirmation for address verification

Example: Can distinguish between 141 Pine St and 151 Pine St by matching
architectural features, landscaping, and house colors.
"""

import logging
import os
import base64
from typing import List, Dict, Optional, Tuple
import requests
from io import BytesIO
from PIL import Image
import googlemaps
from openai import OpenAI
from src.utils.resilience import resilient_google_maps_call, resilient_openai_call, resilience_manager

logger = logging.getLogger(__name__)

class StreetViewMatcher:
    def __init__(self):
        self.google_maps_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.client = None
        self.gmaps = None
        
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            logger.info("Street View Matcher initialized with OpenAI")
        
        if self.google_maps_key:
            self.gmaps = googlemaps.Client(key=self.google_maps_key)
            logger.info("Street View Matcher initialized with Google Maps")
    
    def is_available(self) -> bool:
        """Check if both required APIs are available"""
        return bool(self.google_maps_key and self.openai_api_key)
    
    @resilient_google_maps_call
    def get_street_view_image(self, address: str, size: str = "640x640") -> Optional[bytes]:
        """
        Fetch Street View image for a given address
        """
        if not self.google_maps_key:
            return None
            
        try:
            # First geocode the address to get coordinates
            geocode_result = self.gmaps.geocode(address)
            if not geocode_result:
                logger.warning(f"Could not geocode address: {address}")
                return None
                
            location = geocode_result[0]['geometry']['location']
            lat, lng = location['lat'], location['lng']
            
            # Get Street View image
            streetview_url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"?size={size}"
                f"&location={lat},{lng}"
                f"&fov=90"
                f"&pitch=0"
                f"&key={self.google_maps_key}"
            )
            
            response = requests.get(streetview_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully fetched Street View for: {address}")
                return response.content
            else:
                logger.warning(f"Failed to fetch Street View for {address}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Street View for {address}: {e}")
            return None
    
    @resilient_google_maps_call
    def get_multiple_street_views(self, address: str, angles: List[int] = None) -> List[bytes]:
        """
        Get Street View images from multiple angles for better comparison
        """
        if not self.google_maps_key:
            return []
            
        if angles is None:
            angles = [0, 45, 90, 135]  # Default angles
            
        images = []
        try:
            geocode_result = self.gmaps.geocode(address)
            if not geocode_result:
                return []
                
            location = geocode_result[0]['geometry']['location']
            lat, lng = location['lat'], location['lng']
            
            for angle in angles:
                streetview_url = (
                    f"https://maps.googleapis.com/maps/api/streetview"
                    f"?size=640x640"
                    f"&location={lat},{lng}"
                    f"&heading={angle}"
                    f"&fov=90"
                    f"&pitch=0"
                    f"&key={self.google_maps_key}"
                )
                
                response = requests.get(streetview_url, timeout=10)
                if response.status_code == 200:
                    images.append(response.content)
                    
            logger.info(f"Fetched {len(images)} Street View angles for {address}")
            return images
            
        except Exception as e:
            logger.error(f"Error fetching multiple Street Views: {e}")
            return []
    
    @resilient_openai_call
    def compare_images_with_gpt4(self, airbnb_photos: List[str], streetview_image: bytes, address: str) -> Dict:
        """
        Use GPT-4 Vision to compare Airbnb photos with Street View image
        """
        if not self.client:
            return {"match_score": 0, "confidence": 0, "details": "OpenAI API not available"}
            
        try:
            # Convert Street View image to base64
            streetview_b64 = base64.b64encode(streetview_image).decode('utf-8')
            
            # Prepare first 2 Airbnb photos for comparison (reduced for speed)
            airbnb_images_b64 = []
            for url in airbnb_photos[:2]:  # Reduced from 3 to 2 for speed
                try:
                    response = requests.get(url, timeout=5)  # Reduced timeout
                    if response.status_code == 200:
                        img_b64 = base64.b64encode(response.content).decode('utf-8')
                        airbnb_images_b64.append(img_b64)
                except:
                    continue
            
            if not airbnb_images_b64:
                return {"match_score": 0, "confidence": 0, "details": "No Airbnb photos available"}
            
            # Build the message with multiple images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are comparing property photos to verify if they show the same house.
                            
Compare these Airbnb listing photos with the Google Street View image of {address}.

Focus on:
1. House structure and architecture (roof shape, number of stories, building materials)
2. Distinctive features (porch, deck, stairs, windows configuration)
3. Color scheme and exterior materials
4. Landscaping and surroundings
5. Any unique identifiers

Return a JSON response with:
{{
    "match_score": 0-100 (how likely these show the same property),
    "confidence": 0-100 (how confident you are in this assessment),
    "matching_features": ["list of features that match"],
    "non_matching_features": ["list of features that don't match"],
    "details": "Brief explanation of your assessment"
}}

Be thorough but realistic - Street View and listing photos may be taken at different times/seasons."""
                        }
                    ]
                }
            ]
            
            # Add Airbnb photos
            for i, img_b64 in enumerate(airbnb_images_b64):
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low"  # Changed from high to low for speed
                    }
                })
            
            # Add Street View image
            messages[0]["content"].append({
                "type": "text",
                "text": "Street View image:"
            })
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{streetview_b64}",
                    "detail": "low"  # Reduced for speed
                }
            })
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model
                messages=messages,
                max_tokens=300,  # Reduced tokens for speed
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            try:
                result = json.loads(result_text.strip().replace("```json", "").replace("```", ""))
                logger.info(f"Street View comparison for {address}: score={result.get('match_score', 0)}%")
                return result
            except:
                # Fallback if JSON parsing fails
                return {
                    "match_score": 50,
                    "confidence": 50,
                    "details": result_text
                }
                
        except Exception as e:
            logger.error(f"Error comparing images with GPT-4: {e}")
            return {"match_score": 0, "confidence": 0, "details": str(e)}
    
    def find_best_matching_address(self, airbnb_photos: List[str], candidate_addresses: List[str], 
                                    primary_coords: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Compare Airbnb photos with Street View images of candidate addresses to find best match
        
        Returns:
            Dict with best_address, scores for each address, and comparison details
        """
        if not self.is_available():
            logger.warning("Street View Matcher not available (missing API keys)")
            return {
                "best_address": None,
                "scores": {},
                "details": "Street View comparison unavailable"
            }
        
        if not airbnb_photos or not candidate_addresses:
            return {
                "best_address": None,
                "scores": {},
                "details": "No photos or addresses to compare"
            }
        
        results = {}
        best_score = 0
        best_address = None
        
        # Compare each candidate address
        for address in candidate_addresses[:2]:  # Limit to top 2 to avoid timeout
            logger.info(f"Comparing Street View for: {address}")
            
            # Get Street View image
            streetview_image = self.get_street_view_image(address)
            if not streetview_image:
                results[address] = {
                    "match_score": 0,
                    "confidence": 0,
                    "details": "No Street View available"
                }
                continue
            
            # Compare with Airbnb photos
            comparison = self.compare_images_with_gpt4(airbnb_photos, streetview_image, address)
            results[address] = comparison
            
            # Track best match
            score = comparison.get("match_score", 0)
            if score > best_score:
                best_score = score
                best_address = address
        
        # If primary coordinates provided, also check that location
        if primary_coords and len(results) < 3:
            lat, lng = primary_coords
            primary_address = f"{lat},{lng}"
            
            streetview_url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"?size=640x640"
                f"&location={lat},{lng}"
                f"&fov=90"
                f"&pitch=0"
                f"&key={self.google_maps_key}"
            )
            
            try:
                response = requests.get(streetview_url, timeout=10)
                if response.status_code == 200:
                    comparison = self.compare_images_with_gpt4(airbnb_photos, response.content, 
                                                                f"coordinates {lat:.4f}, {lng:.4f}")
                    results[f"Original coords ({lat:.4f}, {lng:.4f})"] = comparison
            except:
                pass
        
        return {
            "best_address": best_address,
            "best_score": best_score,
            "scores": results,
            "details": f"Compared {len(results)} addresses via Street View"
        }