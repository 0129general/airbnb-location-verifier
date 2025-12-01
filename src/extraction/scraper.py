import re
import json
import logging
import requests
import os
import time
import hashlib
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache
import googlemaps
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from urllib.parse import urlparse, quote_plus
from requests.adapters import HTTPAdapter, Retry
import google_streetview.api
import trafilatura
from nlp.nlp_extractor import extract_nlp_location_data, get_best_address_from_nlp

# Import metrics for tracking external API calls
from core.metrics import external_api_calls, external_api_latency, api_errors

# Import caching functionality
from utils.location_cache import cached_location_data, get_cache

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if Apify is enabled
ENABLE_APIFY = os.environ.get("ENABLE_APIFY", "").lower() in ["true", "1", "yes", "on"]
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")

# Auto-enable Apify if token exists and not explicitly disabled
if not ENABLE_APIFY and APIFY_API_TOKEN and os.environ.get("ENABLE_APIFY", "") != "false":
    ENABLE_APIFY = True

# Import ApifyScraper if enabled
apify_scraper = None
if ENABLE_APIFY and APIFY_API_TOKEN:
    try:
        from extraction.apify_scraper import ApifyScraper
        apify_scraper = ApifyScraper(APIFY_API_TOKEN)
        logger.info("Apify integration ENABLED - will use enhanced data extraction")
    except Exception as e:
        logger.warning(f"Could not initialize Apify scraper: {e}")
        ENABLE_APIFY = False
else:
    if not APIFY_API_TOKEN:
        logger.info("Apify integration DISABLED - APIFY_API_TOKEN not found")
    else:
        logger.info("Apify integration DISABLED - ENABLE_APIFY is set to false")

# Caching configuration
GOOGLE_SEARCH_CACHE_TTL = 60 * 60  # 1 hour TTL for Google search results
GEOCODING_CACHE_TTL = 24 * 60 * 60  # 24 hour TTL for geocoding results

# In-memory caches
_google_search_cache = {}
_geocoding_cache = {}

def _get_cache_key(data: str) -> str:
    """Generate a cache key from input data."""
    return hashlib.md5(data.encode()).hexdigest()

def _get_cached_google_search(address: str) -> Optional[Dict[str, Any]]:
    """Get cached Google search result if available and not expired."""
    cache_key = _get_cache_key(f"google_search:{address}")
    if cache_key in _google_search_cache:
        cached_data, timestamp = _google_search_cache[cache_key]
        if time.time() - timestamp < GOOGLE_SEARCH_CACHE_TTL:
            logger.info(f"Returning cached Google search result for: {address}")
            return cached_data
        else:
            del _google_search_cache[cache_key]
    return None

def _cache_google_search(address: str, result: Dict[str, Any]) -> None:
    """Cache Google search result."""
    cache_key = _get_cache_key(f"google_search:{address}")
    _google_search_cache[cache_key] = (result, time.time())
    # Clean up old entries if cache grows too large
    if len(_google_search_cache) > 100:
        sorted_entries = sorted(_google_search_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_entries[:20]:
            del _google_search_cache[old_key]

# LRU cache for geocoding results (24 hour TTL via maxsize)
@lru_cache(maxsize=256)
def _cached_geocode(address: str) -> Optional[Tuple[float, float]]:
    """
    Cached geocoding function using Nominatim.
    Returns (latitude, longitude) or None if geocoding fails.
    """
    try:
        geolocator = Nominatim(user_agent="airbnb-location-verifier", timeout=10)  # type: ignore
        location = geolocator.geocode(address, exactly_one=True)  # type: ignore[misc]
        if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            logger.info(f"Successfully geocoded address (cached): {address}")
            return (location.latitude, location.longitude)  # type: ignore[union-attr]
    except Exception as e:
        logger.warning(f"Geocoding failed for {address}: {e}")
    return None

# ---- Resilient HTTP session ----
def _session_with_retries() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

# ---- URL canonicalization ----
def canonicalize_airbnb_url(url: str) -> str:
    p = urlparse(url)
    scheme = "https"
    netloc = p.netloc.replace("m.airbnb.", "www.airbnb.")
    return f"{scheme}://{netloc}{p.path}"

# ---- Description extraction ----
def _extract_description_text(soup: BeautifulSoup, html: str) -> str:
    """Extract property description text from the Airbnb listing page."""
    description_text = []
    
    # Method 1: Look for description in meta tags
    for meta in soup.find_all("meta"):
        # Type narrowing for BeautifulSoup elements
        if hasattr(meta, 'get'):
            if meta.get("name") == "description" or meta.get("property") == "og:description":  # type: ignore[union-attr]
                content = meta.get("content", "")  # type: ignore[union-attr]
                if content and len(content) > 50:
                    description_text.append(content)
    
    # Method 2: Look for description in structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            # Type narrowing for BeautifulSoup elements
            script_content = getattr(script, 'string', None)
            data = json.loads(script_content or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    # Look for description field
                    if item.get("description"):
                        description_text.append(str(item["description"]))
        except:
            pass
    
    # Method 3: Look for description in __NEXT_DATA__
    for script in soup.find_all("script"):
        # Type narrowing for BeautifulSoup elements
        if hasattr(script, 'get') and script.get("id") in ("__NEXT_DATA__", "data-state"):  # type: ignore[union-attr]
            try:
                script_content = getattr(script, 'string', None)
                if script_content:
                    data = json.loads(script_content)
                    # Recursive search for description fields
                    def find_descriptions(obj, depth=0):
                        if depth > 10:  # Prevent infinite recursion
                            return []
                        descs = []
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if 'description' in k.lower() and isinstance(v, str) and len(v) > 50:
                                    descs.append(v)
                                elif isinstance(v, (dict, list)):
                                    descs.extend(find_descriptions(v, depth + 1))
                        elif isinstance(obj, list):
                            for item in obj[:100]:  # Limit to first 100 items
                                descs.extend(find_descriptions(item, depth + 1))
                        return descs
                    
                    found_descs = find_descriptions(data)
                    description_text.extend(found_descs[:5])  # Take max 5 descriptions
            except:
                pass
    
    # Method 4: Look for common description patterns in HTML text
    # Look for sections that might contain descriptions
    description_sections = soup.find_all(['div', 'section', 'p'], string=re.compile(
        r'(located|situated|nestled|perfect for|ideal for|features|offers|walking distance|minutes from|close to|near|beach|park|downtown|neighborhood)',
        re.IGNORECASE
    ))
    
    for section in description_sections[:10]:  # Limit to first 10 matches
        text = section.get_text(strip=True)
        if len(text) > 50 and len(text) < 5000:  # Reasonable length for description
            description_text.append(text)
    
    # Combine all found descriptions, removing duplicates
    combined_description = '\n'.join(list(dict.fromkeys(description_text)))
    
    # If no description found, try to extract from visible text
    if not combined_description:
        # Get all text from the page
        all_text = soup.get_text()
        # Look for description-like sections
        matches = re.findall(
            r'(?:located|situated|nestled|perfect|ideal|features|offers|property|home|house|apartment|condo)[^.]{50,1000}',
            all_text,
            re.IGNORECASE
        )
        if matches:
            combined_description = '\n'.join(matches[:5])
    
    logger.info(f"Extracted description text of length: {len(combined_description)}")
    return combined_description[:10000]  # Limit to 10000 chars

# ---- Photo extraction ----
def _extract_photo_urls(soup: BeautifulSoup, html: str) -> list:
    """Extract all photo URLs from the Airbnb listing page."""
    photos = []
    seen_urls = set()
    
    # Filter out non-property images
    exclude_patterns = [
        'UserProfile',
        'search-bar-icons',
        'platform-assets',
        'host-passport',
        'user_profile',
        'avatar'
    ]
    
    def is_property_photo(url):
        """Check if URL is likely a property photo."""
        url_lower = url.lower()
        # Exclude profile and UI images
        for pattern in exclude_patterns:
            if pattern.lower() in url_lower:
                return False
        # Include hosting/property images
        if 'hosting-' in url_lower or '/pictures/' in url_lower or 'prohost' in url_lower:
            return True
        # Include general muscache images that aren't excluded
        if 'muscache.com' in url_lower:
            return True
        return False
    
    # Method 1: Look for images in picture tags
    for picture in soup.find_all("picture"):
        # Type narrowing for BeautifulSoup elements
        if hasattr(picture, 'find_all'):
            for source in picture.find_all("source"):  # type: ignore[union-attr]
                if hasattr(source, 'get'):
                    srcset = source.get("srcset", "")  # type: ignore[union-attr]
                    if isinstance(srcset, str):
                        for url in srcset.split(","):
                            url = url.strip().split(" ")[0]
                            if url and url not in seen_urls and is_property_photo(url):
                                seen_urls.add(url)
                                photos.append(url)
        
        if hasattr(picture, 'find'):
            img = picture.find("img")  # type: ignore[union-attr]
            if img and hasattr(img, 'get'):
                src = img.get("src")  # type: ignore[union-attr]
                if isinstance(src, str) and src not in seen_urls and is_property_photo(src):
                    seen_urls.add(src)
                    photos.append(src)
    
    # Method 2: Look for direct img tags with Airbnb CDN URLs
    for img in soup.find_all("img"):
        if hasattr(img, 'get'):
            src = img.get("src", "")  # type: ignore[union-attr]
            if isinstance(src, str) and ("muscache.com" in src or "airbnb" in src) and is_property_photo(src):
                if src not in seen_urls:
                    seen_urls.add(src)
                    photos.append(src)
    
    # Method 3: Extract from JSON data in scripts - prioritize prohost/hosting URLs
    for script in soup.find_all("script"):
        script_content = getattr(script, 'string', None)
        if script_content:
            # Look specifically for property image URLs
            patterns = [
                r'"(https://[^"]*prohost[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r'"(https://[^"]*pictures/hosting[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r'"(https://[^"]*muscache\.com/im/pictures/[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"'
            ]
            for pattern in patterns:
                urls = re.findall(pattern, script_content)
                for url in urls[:30]:  # Check more URLs
                    # Get high quality version
                    clean_url = url.split("?")[0] + "?im_w=1200"
                    if clean_url not in seen_urls and is_property_photo(clean_url):
                        seen_urls.add(clean_url)
                        photos.append(clean_url)
    
    # Method 4: Look in meta tags
    for meta in soup.find_all("meta", property=re.compile("og:image")):
        if hasattr(meta, 'get'):
            content = meta.get("content")  # type: ignore[union-attr]
            if content and content not in seen_urls and is_property_photo(content):
                seen_urls.add(content)
                photos.append(content)
    
    # Sort photos to prioritize property images
    photos.sort(key=lambda x: (
        'prohost' in x.lower(),
        'hosting-' in x.lower(),
        '/pictures/' in x.lower()
    ), reverse=True)
    
    logger.info(f"Extracted {len(photos)} unique property photo URLs from listing")
    return photos[:15]  # Return max 15 photos to avoid excessive API calls

# ---- Structured data parsers ----
def _extract_from_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    out = {}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            script_content = getattr(tag, 'string', None)
            data = json.loads(script_content or "{}")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            addr = item.get("address") or {}
            if isinstance(addr, dict):
                parts = [addr.get("streetAddress"), addr.get("addressLocality"),
                         addr.get("addressRegion"), addr.get("postalCode"),
                         addr.get("addressCountry")]
                address = ", ".join([x for x in parts if x])
                if address:
                    out["address"] = address
            geo = item.get("geo") or {}
            lat = geo.get("latitude"); lng = geo.get("longitude")
            if lat and lng:
                try:
                    out["latitude"] = float(lat); out["longitude"] = float(lng)
                except Exception:
                    pass
            # Extract photos from structured data
            if item.get("photo") and isinstance(item["photo"], list):
                out["structured_photos"] = [p.get("url") for p in item["photo"] if p.get("url")]
    return out

def _search_dict_for_keys(obj: Any) -> Dict[str, Any]:
    found = {}
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                lk = k.lower()
                if lk in ("lat","latitude"):
                    try: found["latitude"] = float(v)
                    except: pass
                if lk in ("lng","lon","longitude"):
                    try: found["longitude"] = float(v)
                    except: pass
                if "address" in lk and isinstance(v, str):
                    if len(v) > len(found.get("address","")) and ("," in v or " " in v):
                        found["address"] = v.strip()
                # Also look for location names and descriptions
                if "location" in lk and "description" in lk and isinstance(v, str) and len(v) > 10:
                    if "location_description" not in found:
                        found["location_description"] = v.strip()
                if "publicdescription" in lk and isinstance(v, str):
                    if "public_description" not in found:
                        found["public_description"] = v.strip()
                if isinstance(v, (dict, list)): stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return found

def _extract_from_next_data(soup: BeautifulSoup) -> Dict[str, Any]:
    candidates = []
    for tag in soup.find_all("script"):
        if hasattr(tag, 'get'):
            tag_id = tag.get("id")  # type: ignore[union-attr]
            if tag_id in ("__NEXT_DATA__", "data-state"):
                candidates.append(tag)
            elif tag.get("type") == "application/json":  # type: ignore[union-attr]
                script_content = getattr(tag, 'string', None)
                if script_content and len(script_content) > 2000:
                    candidates.append(tag)
    for tag in candidates:
        try:
            script_content = getattr(tag, 'string', None)
            if script_content:
                data = json.loads(script_content)
                found = _search_dict_for_keys(data)
                if found.get("latitude") and found.get("longitude"):
                    return found
        except Exception:
            continue
    return {}

# ---- Geocoding + verification ----
def _reverse_geocode(lat: float, lng: float) -> Optional[str]:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if googlemaps and api_key:
        try:
            gm = googlemaps.Client(key=api_key)
            gr = gm.reverse_geocode((lat, lng))  # type: ignore[attr-defined]
            if gr and isinstance(gr, list) and gr[0].get("formatted_address"):
                return gr[0]["formatted_address"]
        except Exception as e:
            logger.warning(f"Google reverse geocode failed: {e}")
    try:
        geocoder = Nominatim(user_agent="airbnb-geolocator", timeout=10)  # type: ignore
        location = geocoder.reverse((lat, lng))  # type: ignore[misc]
        if location and hasattr(location, 'address'):
            return location.address  # type: ignore[union-attr]
    except Exception as e:
        logger.warning(f"Nominatim reverse geocode failed: {e}")
    return None

def _verify_address_proximity(address: str, lat: float, lng: float, threshold_m=400) -> Tuple[bool, Optional[float]]:
    try:
        geocoder = Nominatim(user_agent="airbnb-geolocator-verify", timeout=10)  # type: ignore
        loc = geocoder.geocode(address)  # type: ignore[misc]
        if not loc:
            return False, None
        if hasattr(loc, 'latitude') and hasattr(loc, 'longitude'):
            dist_m = geodesic((lat, lng), (loc.latitude, loc.longitude)).meters  # type: ignore[union-attr]
            return dist_m <= threshold_m, dist_m
        return False, None
    except Exception as e:
        logger.warning(f"Proximity check failed: {e}")
        return False, None

# Initialize geocoder with a proper user agent (fallback)
geolocator = Nominatim(user_agent="airbnb-geolocation-app")  # type: ignore

# Initialize Google Maps client if API key is available
gmaps = None
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
if GOOGLE_MAPS_API_KEY:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    logger.debug("Google Maps API client initialized")

# Use this for testing when network issues prevent actual scraping
SAFE_MODE = os.environ.get('SAFE_MODE', 'false').lower() == 'true'
logger.info(f"Safe Mode {'ENABLED' if SAFE_MODE else 'DISABLED'}")


@cached_location_data
def get_airbnb_location_data(url):
    """
    Extract location data from an Airbnb listing URL.
    
    This function is now cached to avoid repeated API calls for the same URL.
    Cache TTL is 1 hour by default.
    
    Args:
        url (str): The Airbnb listing URL
        
    Returns:
        dict: A dictionary containing location information including:
            - latitude
            - longitude
            - address
            - verification: proximity check results
            - photos: extracted photo URLs
            - source_url
        
        Returns a dictionary with error message if extraction fails.
    """
    url = canonicalize_airbnb_url(url)
    
    # Try Apify first if enabled
    apify_data = {}
    apify_success = False
    
    if apify_scraper and ENABLE_APIFY:
        try:
            logger.info("Attempting to extract enhanced data via Apify...")
            apify_data = apify_scraper.extract_enhanced_data(url)
            
            if apify_data.get('apify_success'):
                apify_success = True
                logger.info("Successfully retrieved enhanced data from Apify")
                
                # Track API usage
                logger.info("APIFY API USAGE: Successfully scraped 1 listing ($0.00125 estimated cost)")
            else:
                logger.warning("Apify extraction failed, falling back to standard scraping")
        except Exception as e:
            logger.error(f"Apify extraction error: {str(e)}")
            logger.info("Falling back to standard scraping method")
    
    # Standard scraping (either as primary or fallback)
    sess = _session_with_retries()
    
    # Check for 404 before processing
    try:
        resp = sess.get(url, timeout=15)
        
        # Check if listing is no longer available (404 or "listing not found" page)
        if resp.status_code == 404:
            logger.warning(f"Airbnb listing returns 404: {url}")
            return {
                "error": "Listing no longer available",
                "status_code": 404,
                "source_url": url,
                "message": "This Airbnb listing is no longer active or has been removed."
            }
        
        resp.raise_for_status()
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return {
                "error": "Listing no longer available",
                "status_code": 404,
                "source_url": url,
                "message": "This Airbnb listing is no longer active or has been removed."
            }
        raise

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    
    # Also check for soft 404 (page loads but says listing not found)
    title_elem = soup.find("title")
    if title_elem and title_elem.text:
        title_text = title_elem.text.lower()
        if "page not found" in title_text or "this page is unavailable" in title_text or "listing not found" in title_text or "not found" in title_text:
            logger.warning(f"Soft 404 detected for listing: {url}")
            return {
                "error": "Listing no longer available",
                "status_code": 404,
                "source_url": url,
                "message": "This Airbnb listing is no longer active or has been removed."
            }
    
    # Check for generic Airbnb error pages or empty listings
    # Sometimes Airbnb returns coordinates but no real listing data
    has_real_data = False
    
    # Check if we have any meaningful description or property data
    description_elem = soup.find(string=re.compile("About this", re.IGNORECASE))
    if description_elem:
        has_real_data = True
    
    # Check for the room ID in the URL actually having content
    room_id_match = re.search(r'/rooms/(\d+)', url)
    if room_id_match:
        room_id = room_id_match.group(1)
        # Check if this looks like a fake/test ID (very long number like 99999999999)
        if len(room_id) > 10 and room_id == '9' * len(room_id):
            logger.warning(f"Suspicious room ID detected (likely fake): {room_id}")
            return {
                "error": "Listing no longer available", 
                "status_code": 404,
                "source_url": url,
                "message": "This Airbnb listing appears to be invalid or has been removed."
            }

    data: Dict[str, Any] = {}

    # 1) Extract photos for visual analysis
    photos = _extract_photo_urls(soup, html)
    
    # If Apify was successful, merge its enhanced data
    if apify_success and apify_data:
        # Use Apify's location data if available
        if apify_data.get('location'):
            location = apify_data['location']
            if location.get('lat'):
                data['latitude'] = location['lat']
            if location.get('lng'):
                data['longitude'] = location['lng']
            if location.get('address'):
                data['address'] = location['address']
        
        # Merge photos from Apify
        if apify_data.get('photos'):
            apify_photos = apify_data['photos']
            # Add Apify photos that aren't already in our list
            for photo in apify_photos:
                if photo not in photos:
                    photos.append(photo)
        
        # Use the enhanced combined description for better NLP analysis
        description_text = apify_data.get('combined_description', '')
        
        # If no combined description, fall back to regular extraction
        if not description_text:
            description_text = _extract_description_text(soup, html)
    else:
        # 2) Extract description text for NLP analysis (standard method)
        description_text = _extract_description_text(soup, html)
    
    nlp_data = {}
    if description_text:
        logger.info("Extracting NLP location data from description")
        nlp_data = extract_nlp_location_data(description_text)
        logger.debug(f"NLP extraction results: {nlp_data.get('extraction_summary', {})}")
    
    # 3) JSON-LD (precise when present)
    try:
        ld_data = _extract_from_json_ld(soup)
        data.update({k: v for k, v in ld_data.items() if v and k != "structured_photos"})
        # Add structured photos to our photo list
        if ld_data.get("structured_photos"):
            photos.extend([p for p in ld_data["structured_photos"] if p not in photos])
    except Exception as e:
        logger.debug(f"JSON-LD parse error: {e}")

    # 4) __NEXT_DATA__/data-state fallback
    if not data.get("latitude") or not data.get("longitude") or not data.get("address"):
        try:
            nd = _extract_from_next_data(soup)
            for k in ("latitude","longitude","address"):
                if nd.get(k) and not data.get(k):
                    data[k] = nd[k]
        except Exception as e:
            logger.debug(f"NEXT_DATA parse error: {e}")

    # 5) Regex fallback for coords
    if not data.get("latitude") or not data.get("longitude"):
        m = re.search(r'(-?\d{1,2}\.\d{4,7})\s*[, ]\s*(-?\d{1,3}\.\d{4,7})', html)
        if m:
            try:
                data["latitude"] = float(m.group(1))
                data["longitude"] = float(m.group(2))
            except Exception:
                pass

    # 6) Extract location hints from page text if address is generic
    if data.get("address") and len(data.get("address", "").split(",")) <= 2:
        # Address seems generic (like just "Santa Rosa Beach"), try to find more details
        try:
            # Look for location descriptions in the page
            location_hints = []
            
            # Search for common location patterns in the HTML
            patterns = [
                r'(?:located|situated|nestled|found)\s+(?:in|at|on|near)\s+([^\.]{10,60})',
                r'(?:walk|walking|minutes?|blocks?)\s+(?:to|from)\s+([^\.]{10,60})',
                r'(?:close|proximity|near|adjacent)\s+to\s+([^\.]{10,60})',
                r'(?:neighborhood|area|district|zone):\s*([^\.]{10,60})',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                location_hints.extend(matches[:2])  # Take max 2 from each pattern
            
            if location_hints:
                logger.debug(f"Found location hints: {location_hints[:3]}")
                # Store hints for potential use
                data["location_hints"] = location_hints[:5]
        except Exception as e:
            logger.debug(f"Error extracting location hints: {e}")
    
    # 7) Use NLP data to enhance address if needed
    if nlp_data and nlp_data.get('overall_confidence', 0) > 60:
        # Try to get a better address from NLP data
        nlp_address = get_best_address_from_nlp(nlp_data, data.get("address"))
        if nlp_address and nlp_address != data.get("address"):
            # Verify the NLP-enhanced address is reasonable
            if data.get("latitude") and data.get("longitude") and nlp_address:
                ok, dist = _verify_address_proximity(nlp_address, data["latitude"], data["longitude"], 1000)
                if ok:
                    data["original_address"] = data.get("address")
                    data["address"] = nlp_address
                    data["address_source"] = "nlp_enhanced"
                    logger.info(f"Enhanced address from NLP extraction: {nlp_address}")
    
    # 8) Reverse geocode if coords exist but address missing or generic
    if data.get("latitude") and data.get("longitude"):
        if not data.get("address") or len(data.get("address", "").split(",")) <= 2:
            addr = _reverse_geocode(data["latitude"], data["longitude"])
            if addr and len(addr.split(",")) > len(data.get("address", "").split(",")):
                # Reverse geocoded address has more detail
                data["original_listed_address"] = data.get("address")
                data["address"] = addr
                data["address_source"] = "reverse_geocoded"
                logger.info(f"Enhanced address from reverse geocoding: {addr}")

    # 9) Proximity verification
    verification: Dict[str, Any] = {"proximity_ok": None, "distance_m": None}
    if data.get("address") and data.get("latitude") and data.get("longitude"):
        ok, dist = _verify_address_proximity(data["address"], data["latitude"], data["longitude"])
        verification["proximity_ok"] = ok
        verification["distance_m"] = round(dist, 1) if dist is not None and isinstance(dist, (int, float)) else None

    # 10) Sanity check
    if data.get("latitude") == 0 or data.get("longitude") == 0:
        data.pop("latitude", None); data.pop("longitude", None)

    # Prepare the result dictionary
    result = {
        "address": data.get("address"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "verification": verification,
        "photos": photos[:10],  # Return max 10 photos for initial display
        "source_url": url,
        "nlp_extraction": nlp_data,  # Include NLP extraction results
        "description_text": description_text[:500] if description_text else None,  # Include first 500 chars of description
        "address_source": data.get("address_source", "original"),  # Track address source
        "original_address": data.get("original_address"),  # Original address if enhanced
    }
    
    # If Apify was successful, add the enhanced data
    if apify_success and apify_data:
        result["apify_enhanced"] = True
        result["apify_data"] = {
            "full_description": apify_data.get("full_description", ""),
            "space_description": apify_data.get("space_description", ""),
            "neighborhood_overview": apify_data.get("neighborhood_overview", ""),
            "transit_info": apify_data.get("transit_info", ""),
            "getting_around": apify_data.get("getting_around", ""),
            "host_interaction": apify_data.get("host_interaction", ""),
            "amenities": apify_data.get("amenities", []),
            "property_type": apify_data.get("property_type", ""),
            "room_type": apify_data.get("room_type", ""),
            "rating": apify_data.get("rating"),
            "reviews_count": apify_data.get("reviews_count", 0),
            "all_photos": apify_data.get("photos", []),  # All photos from Apify
        }
        # Use the full description for analysis if available
        if apify_data.get("combined_description"):
            result["full_description_text"] = apify_data["combined_description"]
        
        logger.info(f"Enhanced data from Apify included: {len(result['apify_data'].get('amenities', []))} amenities, {len(result['apify_data'].get('all_photos', []))} photos")
    else:
        result["apify_enhanced"] = False
    
    return result


def extract_from_json_ld(soup):
    """Extract location data from JSON-LD structured data - kept for compatibility."""
    # This function is kept for backward compatibility but not used in the new implementation
    # The new _extract_from_json_ld function above handles this functionality
    return _extract_from_json_ld(soup)


def extract_from_inline_js(soup):
    """Extract location data from inline JavaScript - kept for compatibility."""
    # This function is kept for backward compatibility but not used in the new implementation
    return {}


def extract_from_map_iframe(soup):
    """Extract location data from map iframe - kept for compatibility."""
    # This function is kept for backward compatibility but not used in the new implementation
    return {}


def get_google_search_results(address):
    """Generate Google search URLs for an address with caching."""
    if not address:
        return {}
    
    # Check cache first
    cached_result = _get_cached_google_search(address)
    if cached_result:
        return cached_result
    
    # Generate URLs
    q = quote_plus(address)
    result = {
        "search_url": f"https://www.google.com/search?q={q}",
        "maps_search_url": f"https://www.google.com/maps/search/?api=1&query={q}",
        "encoded_address": q
    }
    
    # Cache the result
    _cache_google_search(address, result)
    
    return result


def extract_location_hints_from_text(soup):
    """Extract location hints from page text - kept for compatibility."""
    return ""


def get_street_view_metadata(lat, lng):
    """
    Get Google Street View metadata for given coordinates.
    
    Args:
        lat (float): Latitude
        lng (float): Longitude
        
    Returns:
        dict: Street View metadata including:
            - available: bool, whether street view is available
            - status: API response status
            - panorama_id: ID of the street view panorama
            - date: Date of the street view image
            - heading: Camera heading for optimal view
            - static_image_url: URL for static street view image
    """
    metadata = {
        'available': False,
        'status': 'NOT_CHECKED',
        'panorama_id': None,
        'date': None,
        'heading': None,
        'static_image_url': None
    }
    
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not found - Street View metadata unavailable")
        metadata['status'] = 'NO_API_KEY'
        return metadata
    
    # Track Street View API call start time
    api_start_time = time.time()
    
    try:
        # Use the initialized gmaps client
        if not gmaps:
            logger.error("Google Maps client not initialized")
            metadata['status'] = 'CLIENT_ERROR'
            return metadata
        
        # Check if street view is available using the street_view API
        # We'll use the metadata endpoint to check availability
        location = f"{lat},{lng}"
        
        # Make a metadata request to check if street view is available
        # The Google Maps Python client doesn't have direct street view metadata support,
        # so we'll make a direct API request
        import requests
        base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            'location': location,
            'key': GOOGLE_MAPS_API_KEY,
            'source': 'outdoor',  # Prefer outdoor panoramas
            'radius': 50  # Search within 50 meters
        }
        
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        
        # Track Street View API call metrics
        api_latency = time.time() - api_start_time
        external_api_latency.labels(service='google_streetview', endpoint='metadata').observe(api_latency)
        
        if data.get('status') == 'OK':
            metadata['available'] = True
            metadata['status'] = 'OK'
            metadata['panorama_id'] = data.get('pano_id')
            metadata['date'] = data.get('date')
            external_api_calls.labels(service='google_streetview', endpoint='metadata', status='success').inc()
            
            # Calculate optimal heading (towards the property)
            # For now, we'll use a default heading of 0, but in production
            # you might calculate this based on property entrance
            metadata['heading'] = 0
            
            # Generate static street view image URL
            static_base_url = "https://maps.googleapis.com/maps/api/streetview"
            static_params = {
                'size': '600x400',
                'location': location,
                'heading': metadata['heading'],
                'pitch': '0',
                'fov': '90',
                'key': GOOGLE_MAPS_API_KEY,
                'source': 'outdoor'
            }
            
            # Build the static image URL
            static_url_parts = []
            for k, v in static_params.items():
                static_url_parts.append(f"{k}={v}")
            
            metadata['static_image_url'] = f"{static_base_url}?{'&'.join(static_url_parts)}"
            
            logger.info(f"Street View available at {lat},{lng} - Panorama ID: {metadata['panorama_id']}")
            
        elif data.get('status') == 'ZERO_RESULTS':
            metadata['status'] = 'NOT_AVAILABLE'
            logger.info(f"No Street View available at {lat},{lng}")
            external_api_calls.labels(service='google_streetview', endpoint='metadata', status='not_found').inc()
            
        else:
            metadata['status'] = data.get('status', 'UNKNOWN')
            logger.warning(f"Street View API returned status: {metadata['status']}")
            external_api_calls.labels(service='google_streetview', endpoint='metadata', status='error').inc()
            
    except requests.exceptions.Timeout:
        metadata['status'] = 'TIMEOUT'
        logger.error("Street View API request timed out")
        api_latency = time.time() - api_start_time
        external_api_latency.labels(service='google_streetview', endpoint='metadata').observe(api_latency)
        external_api_calls.labels(service='google_streetview', endpoint='metadata', status='timeout').inc()
        api_errors.labels(service='google_streetview', error_type='timeout').inc()
        
    except Exception as e:
        metadata['status'] = 'ERROR'
        logger.error(f"Error getting Street View metadata: {e}")
        api_latency = time.time() - api_start_time
        external_api_latency.labels(service='google_streetview', endpoint='metadata').observe(api_latency)
        external_api_calls.labels(service='google_streetview', endpoint='metadata', status='error').inc()
        api_errors.labels(service='google_streetview', error_type=type(e).__name__).inc()
    
    return metadata


def get_google_images_for_address(address, limit=5):
    """Get Google images for an address - kept for compatibility."""
    return []


def search_realtorcom_listings(address):
    """Search Realtor.com listings - kept for compatibility."""
    return {}