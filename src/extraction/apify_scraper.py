"""
Apify Scraper Module for Enhanced Airbnb Data Extraction
Uses the tri_angle/airbnb-rooms-urls-scraper actor for individual listing data
"""

import os
import re
import json
import time
import logging
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from src.utils.resilience import resilient_apify_call, resilience_manager

# Configure logging
logger = logging.getLogger(__name__)

class ApifyScraper:
    """
    Scraper class that uses Apify's Airbnb scraper actor
    to extract comprehensive listing data.
    """
    
    # Actor ID for the tri_angle/airbnb-rooms-urls-scraper (designed for individual room URLs)
    ACTOR_ID = "tri_angle/airbnb-rooms-urls-scraper"
    APIFY_API_URL = "https://api.apify.com/v2"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Apify scraper with API token.
        
        Args:
            api_token: Apify API token. If None, will look for APIFY_API_TOKEN env var.
        """
        self.api_token = api_token or os.environ.get("APIFY_API_TOKEN")
        if not self.api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables")
        
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info("ApifyScraper initialized with API token")
    
    def extract_listing_id(self, url: str) -> Optional[str]:
        """
        Extract the Airbnb listing ID from a URL.
        
        Args:
            url: The Airbnb listing URL
            
        Returns:
            The listing ID or None if not found
        """
        # Match pattern /rooms/12345 or /rooms/plus/12345
        match = re.search(r'/rooms/(?:plus/)?(\d+)', url)
        if match:
            return match.group(1)
        return None
    
    @resilient_apify_call
    def run_actor(self, listing_url: str, max_wait_seconds: int = 25) -> Optional[Dict[str, Any]]:
        """
        Run the Apify Airbnb scraper actor and wait for results.
        
        Args:
            listing_url: The Airbnb listing URL to scrape
            max_wait_seconds: Maximum time to wait for actor to complete
            
        Returns:
            The scraped data or None if failed
        """
        try:
            listing_id = self.extract_listing_id(listing_url)
            if not listing_id:
                logger.error(f"Could not extract listing ID from URL: {listing_url}")
                return None
            
            # Input for the actor (tri_angle/airbnb-rooms-urls-scraper expects startUrls)
            actor_input = {
                "startUrls": [{"url": listing_url}]
            }
            
            # Start the actor run (need to URL-encode the actor ID properly)
            actor_id_encoded = self.ACTOR_ID.replace("/", "~")
            run_url = f"{self.APIFY_API_URL}/acts/{actor_id_encoded}/runs?token={self.api_token}"
            
            logger.info(f"Starting Apify actor for listing ID: {listing_id}")
            response = requests.post(run_url, json=actor_input, headers=self.headers, timeout=5)
            
            if response.status_code != 201:
                logger.error(f"Failed to start actor: {response.status_code} - {response.text}")
                return None
            
            run_data = response.json()
            run_id = run_data.get("data", {}).get("id")
            
            if not run_id:
                logger.error("No run ID returned from Apify")
                return None
            
            logger.info(f"Actor run started with ID: {run_id}")
            
            # Wait for the actor to complete
            return self._wait_for_run(run_id, max_wait_seconds)
            
        except Exception as e:
            logger.error(f"Error running Apify actor: {str(e)}")
            return None
    
    def _wait_for_run(self, run_id: str, max_wait_seconds: int) -> Optional[Dict[str, Any]]:
        """
        Wait for an Apify actor run to complete and return results.
        
        Args:
            run_id: The ID of the actor run
            max_wait_seconds: Maximum time to wait
            
        Returns:
            The scraped data or None if failed/timeout
        """
        status_url = f"{self.APIFY_API_URL}/actor-runs/{run_id}?token={self.api_token}"
        dataset_url = f"{self.APIFY_API_URL}/actor-runs/{run_id}/dataset/items?token={self.api_token}"
        
        start_time = time.time()
        check_interval = 2  # Start with 2 second intervals
        
        while time.time() - start_time < max_wait_seconds:
            try:
                # Check run status
                response = requests.get(status_url, timeout=5)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get run status: {response.status_code}")
                    return None
                
                run_data = response.json().get("data", {})
                status = run_data.get("status")
                
                logger.debug(f"Actor run status: {status}")
                
                if status == "SUCCEEDED":
                    # Get the results
                    logger.info("Actor run completed successfully, fetching results...")
                    results_response = requests.get(dataset_url, timeout=5)
                    
                    if results_response.status_code == 200:
                        results = results_response.json()
                        if results and len(results) > 0:
                            return results[0]  # Return first result
                        else:
                            logger.warning("Actor completed but returned no results")
                            return None
                    else:
                        logger.error(f"Failed to fetch results: {results_response.status_code}")
                        return None
                
                elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                    logger.error(f"Actor run failed with status: {status}")
                    return None
                
                # Still running, wait and check again
                time.sleep(min(check_interval, 10))  # Cap at 10 seconds
                check_interval *= 1.5  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error checking run status: {str(e)}")
                return None
        
        logger.error(f"Actor run timed out after {max_wait_seconds} seconds")
        return None
    
    def extract_enhanced_data(self, listing_url: str) -> Dict[str, Any]:
        """
        Extract enhanced data from an Airbnb listing using Apify.
        
        Args:
            listing_url: The Airbnb listing URL
            
        Returns:
            Dictionary containing enhanced listing data or empty dict if failed
        """
        try:
            logger.info(f"Using Apify to extract enhanced data from: {listing_url}")
            
            # Run the actor and get results
            raw_data = self.run_actor(listing_url)
            
            if not raw_data:
                logger.warning("Apify actor returned no data")
                return {}
            
            # Extract all the enhanced fields from the new actor format
            # The airbnb-rooms-urls-scraper returns different field names
            
            # Handle coordinates safely
            coordinates = raw_data.get('coordinates', {})
            if isinstance(coordinates, dict):
                lat = coordinates.get('lat')
                lng = coordinates.get('lng') 
            else:
                lat = None
                lng = None
                
            # Handle location safely
            location_data = raw_data.get('location', {})
            if not isinstance(location_data, dict):
                location_data = {}
            
            enhanced_data = {
                'apify_success': True,
                'listing_id': raw_data.get('id'),
                'title': raw_data.get('title', raw_data.get('sharingConfigTitle', '')),
                'full_description': raw_data.get('description', ''),
                'html_description': raw_data.get('htmlDescription', ''),
                'sub_description': raw_data.get('subDescription', ''),
                'location_descriptions': raw_data.get('locationDescriptions', []),
                'location_subtitle': raw_data.get('locationSubtitle', ''),
                'amenities': raw_data.get('amenities', []),
                'house_rules': raw_data.get('houseRules', ''),
                'location': {
                    'lat': lat,
                    'lng': lng,
                    'address': location_data.get('address', ''),
                    'city': location_data.get('city', ''),
                    'country': location_data.get('country', ''),
                    'full_location': location_data,
                },
                'photos': self._extract_photos(raw_data),
                'host': raw_data.get('host', {}),
                'property_type': raw_data.get('propertyType', ''),
                'room_type': raw_data.get('roomType', ''),
                'rating': raw_data.get('rating'),
                'price': raw_data.get('price'),
                'highlights': raw_data.get('highlights', []),
                'breadcrumbs': raw_data.get('breadcrumbs', []),
                'raw_data': raw_data  # Keep raw data for debugging
            }
            
            # Combine all text descriptions for AI analysis
            description_parts = []
            
            # Add main descriptions
            for key in ['full_description', 'html_description', 'sub_description', 'location_subtitle']:
                if enhanced_data.get(key):
                    description_parts.append(str(enhanced_data[key]))
            
            # Add location descriptions (it's a list)
            location_descs = enhanced_data.get('location_descriptions', [])
            if location_descs:
                if isinstance(location_descs, list):
                    description_parts.extend([str(desc) for desc in location_descs if desc])
                else:
                    description_parts.append(str(location_descs))
            
            # Add highlights if available
            highlights = enhanced_data.get('highlights', [])
            if highlights and isinstance(highlights, list):
                description_parts.extend([str(h) for h in highlights if h])
            
            enhanced_data['combined_description'] = '\n\n'.join(description_parts)
            
            # Log extraction success
            logger.info(f"Successfully extracted enhanced data via Apify:")
            logger.info(f"  - Description length: {len(enhanced_data.get('full_description', ''))}")
            logger.info(f"  - HTML description length: {len(enhanced_data.get('html_description', ''))}")
            logger.info(f"  - Location descriptions count: {len(enhanced_data.get('location_descriptions', []))}")
            logger.info(f"  - Combined description length: {len(enhanced_data.get('combined_description', ''))}")
            logger.info(f"  - Amenities count: {len(enhanced_data.get('amenities', []))}")
            logger.info(f"  - Photos count: {len(enhanced_data.get('photos', []))}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error extracting enhanced data via Apify: {str(e)}")
            return {'apify_success': False, 'error': str(e)}
    
    def _extract_photos(self, raw_data: Dict[str, Any]) -> List[str]:
        """
        Extract photo URLs from the raw Apify data.
        
        Args:
            raw_data: Raw data from Apify
            
        Returns:
            List of photo URLs
        """
        photos = []
        
        # Try different possible photo field names
        photo_fields = ['photos', 'images', 'pictures', 'xl_picture_urls', 'picture_urls']
        
        for field in photo_fields:
            if field in raw_data:
                field_data = raw_data[field]
                if isinstance(field_data, list):
                    photos.extend([str(url) for url in field_data if url])
                break
        
        # Also check for photos in a nested structure
        if not photos and 'listing' in raw_data:
            listing = raw_data['listing']
            if isinstance(listing, dict):
                for field in photo_fields:
                    if field in listing:
                        field_data = listing[field]
                        if isinstance(field_data, list):
                            photos.extend([str(url) for url in field_data if url])
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_photos = []
        for photo in photos:
            if photo not in seen:
                seen.add(photo)
                unique_photos.append(photo)
        
        return unique_photos[:50]  # Limit to 50 photos to avoid overwhelming the analysis