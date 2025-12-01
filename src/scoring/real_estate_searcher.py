"""
Real Estate Search Module for cross-referencing Airbnb properties
with Zillow and Realtor.com listings.

Enhanced with:
- Rate limiting with service-specific rules
- Response caching (24 hours)
- Circuit breaker pattern
- Fallback to vision/NLP data when services are blocked
- Improved error handling and logging
"""

import re
import json
import time
import logging
import hashlib
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote_plus, urljoin
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Lock
from enum import Enum

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for rate limiting and timeouts
RATE_LIMIT_DELAY = 0.5  # Reduced delay for parallel execution
MAX_RETRIES = 2  # Reduced retries for faster failures
REQUEST_TIMEOUT = 5  # Reduced timeout as per requirements
SEARCH_TIMEOUT = 8  # Overall timeout for each search service

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.real_estate_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Service status enum
class ServiceStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_BROKEN = "circuit_broken"

@dataclass
class ServiceState:
    """Track state of each real estate service."""
    status: ServiceStatus = ServiceStatus.AVAILABLE
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    backoff_minutes: float = 5.0  # For exponential backoff
    circuit_broken_until: Optional[datetime] = None
    rate_limited_until: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0

@dataclass
class CachedResult:
    """Cached search result with metadata."""
    data: Any
    timestamp: datetime
    address_normalized: str
    service: str

@dataclass
class PropertyDetails:
    """Data class for property information."""
    address: str
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    year_built: Optional[int] = None
    price: Optional[str] = None
    property_type: Optional[str] = None
    lot_size: Optional[str] = None
    listing_url: Optional[str] = None
    source: Optional[str] = None
    confidence_score: float = 0.0
    raw_data: Optional[Dict] = None
    from_cache: bool = False


class RateLimiter:
    """Manages rate limiting and circuit breaker for services."""
    
    def __init__(self):
        self.states: Dict[str, ServiceState] = {
            "zillow": ServiceState(),
            "realtor": ServiceState()
        }
        self.lock = Lock()
        self._load_state()
    
    def _get_state_file(self) -> str:
        """Get path to persistent state file."""
        return os.path.join(CACHE_DIR, 'service_states.json')
    
    def _load_state(self):
        """Load service states from disk."""
        try:
            state_file = self._get_state_file()
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    for service, state_data in data.items():
                        if service in self.states:
                            state = ServiceState()
                            state.status = ServiceStatus(state_data.get('status', 'available'))
                            state.consecutive_failures = state_data.get('consecutive_failures', 0)
                            state.backoff_minutes = state_data.get('backoff_minutes', 5.0)
                            state.total_failures = state_data.get('total_failures', 0)
                            state.total_successes = state_data.get('total_successes', 0)
                            
                            # Parse datetime fields
                            if state_data.get('last_failure_time'):
                                state.last_failure_time = datetime.fromisoformat(state_data['last_failure_time'])
                            if state_data.get('circuit_broken_until'):
                                state.circuit_broken_until = datetime.fromisoformat(state_data['circuit_broken_until'])
                            if state_data.get('rate_limited_until'):
                                state.rate_limited_until = datetime.fromisoformat(state_data['rate_limited_until'])
                            
                            self.states[service] = state
                            
                logger.info(f"Loaded service states: Zillow={self.states['zillow'].status.value}, "
                           f"Realtor={self.states['realtor'].status.value}")
        except Exception as e:
            logger.warning(f"Could not load service states: {e}")
    
    def _save_state(self):
        """Save service states to disk."""
        try:
            state_file = self._get_state_file()
            data = {}
            for service, state in self.states.items():
                data[service] = {
                    'status': state.status.value,
                    'consecutive_failures': state.consecutive_failures,
                    'backoff_minutes': state.backoff_minutes,
                    'total_failures': state.total_failures,
                    'total_successes': state.total_successes,
                }
                if state.last_failure_time:
                    data[service]['last_failure_time'] = state.last_failure_time.isoformat()
                if state.circuit_broken_until:
                    data[service]['circuit_broken_until'] = state.circuit_broken_until.isoformat()
                if state.rate_limited_until:
                    data[service]['rate_limited_until'] = state.rate_limited_until.isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save service states: {e}")
    
    def can_make_request(self, service: str) -> Tuple[bool, str]:
        """Check if we can make a request to the service."""
        with self.lock:
            state = self.states.get(service)
            if not state:
                return True, "OK"
            
            now = datetime.now()
            
            # Check circuit breaker
            if state.status == ServiceStatus.CIRCUIT_BROKEN:
                if state.circuit_broken_until and now < state.circuit_broken_until:
                    remaining = (state.circuit_broken_until - now).total_seconds() / 60
                    msg = f"Circuit breaker active for {service}, retry in {remaining:.1f} minutes"
                    logger.warning(msg)
                    return False, msg
                else:
                    # Reset circuit breaker
                    logger.info(f"Resetting circuit breaker for {service}")
                    state.status = ServiceStatus.AVAILABLE
                    state.consecutive_failures = 0
                    state.circuit_broken_until = None
                    self._save_state()
            
            # Check rate limiting
            if state.status == ServiceStatus.RATE_LIMITED:
                if state.rate_limited_until and now < state.rate_limited_until:
                    remaining = (state.rate_limited_until - now).total_seconds() / 60
                    msg = f"{service} is rate limited, retry in {remaining:.1f} minutes"
                    logger.warning(msg)
                    return False, msg
                else:
                    # Reset rate limiting
                    logger.info(f"Rate limit expired for {service}")
                    state.status = ServiceStatus.AVAILABLE
                    state.rate_limited_until = None
                    state.backoff_minutes = 5.0  # Reset backoff
                    self._save_state()
            
            return True, "OK"
    
    def record_failure(self, service: str, status_code: Optional[int] = None, error: Optional[str] = None):
        """Record a failed request."""
        with self.lock:
            state = self.states.get(service)
            if not state:
                return
            
            state.consecutive_failures += 1
            state.total_failures += 1
            state.last_failure_time = datetime.now()
            
            # Handle specific status codes
            if status_code == 403 and service == "zillow":
                # Zillow 403: Block for 1 hour
                state.status = ServiceStatus.RATE_LIMITED
                state.rate_limited_until = datetime.now() + timedelta(hours=1)
                logger.error(f"Zillow returned 403 Forbidden - blocking for 1 hour")
                
            elif status_code == 429 and service == "realtor":
                # Realtor 429: Exponential backoff
                state.status = ServiceStatus.RATE_LIMITED
                state.rate_limited_until = datetime.now() + timedelta(minutes=state.backoff_minutes)
                logger.error(f"Realtor returned 429 Too Many Requests - backing off for {state.backoff_minutes} minutes")
                state.backoff_minutes = min(state.backoff_minutes * 2, 60)  # Cap at 60 minutes
                
            # Check for circuit breaker trigger
            if state.consecutive_failures >= 3:
                state.status = ServiceStatus.CIRCUIT_BROKEN
                state.circuit_broken_until = datetime.now() + timedelta(minutes=30)
                logger.error(f"Circuit breaker triggered for {service} after {state.consecutive_failures} failures")
            
            self._save_state()
    
    def record_success(self, service: str):
        """Record a successful request."""
        with self.lock:
            state = self.states.get(service)
            if not state:
                return
            
            state.consecutive_failures = 0
            state.total_successes += 1
            
            # Reset backoff on success
            if state.status == ServiceStatus.AVAILABLE:
                state.backoff_minutes = 5.0
            
            self._save_state()
    
    def get_status(self) -> Dict[str, Dict]:
        """Get current status of all services."""
        status = {}
        with self.lock:
            for service, state in self.states.items():
                status[service] = {
                    'status': state.status.value,
                    'consecutive_failures': state.consecutive_failures,
                    'total_failures': state.total_failures,
                    'total_successes': state.total_successes,
                }
                if state.rate_limited_until:
                    status[service]['rate_limited_until'] = state.rate_limited_until.isoformat()
                if state.circuit_broken_until:
                    status[service]['circuit_broken_until'] = state.circuit_broken_until.isoformat()
        return status


class ResponseCache:
    """Manages caching of search results."""
    
    def __init__(self, cache_hours: int = 24):
        self.cache_hours = cache_hours
        self.lock = Lock()
    
    def _get_cache_key(self, address: str, service: str) -> str:
        """Generate cache key from normalized address."""
        # Normalize address for caching
        normalized = self._normalize_address(address).lower()
        key_str = f"{service}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for cache key."""
        if not address:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = " ".join(address.lower().split())
        
        # Remove punctuation except for numbers and letters
        normalized = re.sub(r'[^\w\s#-]', '', normalized)
        
        return normalized.strip()
    
    def _get_cache_file(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(CACHE_DIR, f"cache_{cache_key}.pkl")
    
    def get(self, address: str, service: str) -> Optional[List[PropertyDetails]]:
        """Get cached results if available and not expired."""
        cache_key = self._get_cache_key(address, service)
        cache_file = self._get_cache_file(cache_key)
        
        with self.lock:
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached: CachedResult = pickle.load(f)
                    
                    # Check expiration
                    age_hours = (datetime.now() - cached.timestamp).total_seconds() / 3600
                    if age_hours < self.cache_hours:
                        logger.info(f"Cache hit for {service} search: {address} (age: {age_hours:.1f} hours)")
                        # Mark results as from cache
                        for prop in cached.data:
                            prop.from_cache = True
                        return cached.data
                    else:
                        logger.info(f"Cache expired for {service}: {address} (age: {age_hours:.1f} hours)")
                        os.remove(cache_file)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set(self, address: str, service: str, data: List[PropertyDetails]):
        """Cache search results."""
        cache_key = self._get_cache_key(address, service)
        cache_file = self._get_cache_file(cache_key)
        
        with self.lock:
            try:
                cached = CachedResult(
                    data=data,
                    timestamp=datetime.now(),
                    address_normalized=self._normalize_address(address),
                    service=service
                )
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached, f)
                logger.info(f"Cached {len(data)} results for {service}: {address}")
            except Exception as e:
                logger.error(f"Cache write error: {e}")
    
    def clear_old_cache(self):
        """Remove expired cache files."""
        with self.lock:
            try:
                now = datetime.now()
                for filename in os.listdir(CACHE_DIR):
                    if filename.startswith('cache_') and filename.endswith('.pkl'):
                        filepath = os.path.join(CACHE_DIR, filename)
                        try:
                            with open(filepath, 'rb') as f:
                                cached: CachedResult = pickle.load(f)
                            age_hours = (now - cached.timestamp).total_seconds() / 3600
                            if age_hours > self.cache_hours:
                                os.remove(filepath)
                                logger.info(f"Removed expired cache file: {filename}")
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Error clearing old cache: {e}")


class RealEstateSearcher:
    """Main class for searching real estate listings across multiple platforms."""
    
    def __init__(self):
        """Initialize the searcher with rate limiting and caching."""
        self.session = self._create_session()
        self.last_request_time = 0
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache(cache_hours=24)
        
        # Clean up old cache on startup
        self.cache.clear_old_cache()
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic and proper headers."""
        session = requests.Session()
        retries = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]  # Don't retry on 403/429
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Set realistic browser headers
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        
        return session
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, service: str, **kwargs) -> Optional[requests.Response]:
        """Make a rate-limited request with error handling and circuit breaker."""
        # Check if service is available
        can_request, reason = self.rate_limiter.can_make_request(service)
        if not can_request:
            logger.warning(f"Skipping request to {service}: {reason}")
            return None
        
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            
            # Check for rate limiting responses
            if response.status_code == 403:
                self.rate_limiter.record_failure(service, 403)
                logger.error(f"{service} returned 403 Forbidden")
                return None
            elif response.status_code == 429:
                self.rate_limiter.record_failure(service, 429)
                logger.error(f"{service} returned 429 Too Many Requests")
                return None
            
            response.raise_for_status()
            self.rate_limiter.record_success(service)
            return response
            
        except requests.exceptions.RequestException as e:
            self.rate_limiter.record_failure(service, error=str(e))
            logger.error(f"Request failed for {url}: {str(e)}")
            return None
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for comparison."""
        if not address:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = " ".join(address.lower().split())
        
        # Common abbreviation replacements
        replacements = {
            " street": " st",
            " avenue": " ave",
            " road": " rd",
            " boulevard": " blvd",
            " drive": " dr",
            " lane": " ln",
            " court": " ct",
            " place": " pl",
            " circle": " cir",
            " north ": " n ",
            " south ": " s ",
            " east ": " e ",
            " west ": " w ",
            " apartment": " apt",
            " suite": " ste",
            " unit": " #",
            " mountain ": " mt ",
            " mount ": " mt ",
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove punctuation except for numbers and letters
        normalized = re.sub(r'[^\w\s#-]', '', normalized)
        
        return normalized.strip()
    
    def _calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity score between two addresses using fuzzy matching."""
        if not addr1 or not addr2:
            return 0.0
        
        # Normalize both addresses
        norm1 = self._normalize_address(addr1)
        norm2 = self._normalize_address(addr2)
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Check for specific components match
        components_score = 0
        
        # Extract house number
        num1 = re.findall(r'^\d+', norm1)
        num2 = re.findall(r'^\d+', norm2)
        if num1 and num2 and num1[0] == num2[0]:
            components_score += 0.3
        
        # Extract street name (simplified)
        street1_match = re.search(r'\d+\s+(\w+)', norm1)
        street2_match = re.search(r'\d+\s+(\w+)', norm2)
        if street1_match and street2_match:
            if street1_match.group(1) == street2_match.group(1):
                components_score += 0.3
        
        # Check zip code
        zip1 = re.findall(r'\b\d{5}\b', norm1)
        zip2 = re.findall(r'\b\d{5}\b', norm2)
        if zip1 and zip2 and zip1[-1] == zip2[-1]:
            components_score += 0.2
        
        # Combine scores (weighted average)
        final_score = (similarity * 0.6) + min(components_score, 0.4)
        
        return min(final_score, 1.0)
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text like '3 beds' or '2.5 baths'."""
        if not text:
            return None
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            try:
                value = float(match.group(1))
                return int(value) if value.is_integer() else value
            except ValueError:
                pass
        return None
    
    def search_zillow(self, address: str, features: Optional[Dict] = None) -> List[PropertyDetails]:
        """
        Search Zillow for properties matching the given address.
        
        Args:
            address: Property address to search for
            features: Optional dict with property features for verification
            
        Returns:
            List of PropertyDetails objects for matching properties
        """
        # Check cache first
        cached_results = self.cache.get(address, "zillow")
        if cached_results is not None:
            return cached_results
        
        properties = []
        
        try:
            # Check if service is available
            can_request, reason = self.rate_limiter.can_make_request("zillow")
            if not can_request:
                logger.warning(f"Zillow search skipped: {reason}")
                return properties
            
            # Format address for Zillow search
            search_query = quote_plus(address)
            search_url = f"https://www.zillow.com/homes/{search_query}_rb/"
            
            logger.info(f"Searching Zillow for: {address}")
            
            response = self._make_request(search_url, "zillow")
            if not response:
                logger.warning("Failed to get Zillow search results")
                return properties
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple selectors as Zillow changes their HTML structure
            selectors = [
                'article[data-test="property-card"]',
                'div[class*="list-card"]',
                'div[class*="property-card"]',
                'article[class*="StyledPropertyCard"]'
            ]
            
            property_cards = []
            for selector in selectors:
                property_cards = soup.select(selector)
                if property_cards:
                    break
            
            if not property_cards:
                # Try to find properties in JSON data
                scripts = soup.find_all('script', type='application/json')
                for script in scripts:
                    try:
                        data = json.loads(script.string)
                        properties_from_json = self._extract_zillow_from_json(data, address)
                        if properties_from_json:
                            properties.extend(properties_from_json)
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            # Process property cards from HTML
            for card in property_cards[:10]:  # Limit to first 10 results
                try:
                    prop = self._parse_zillow_card(card, address)
                    if prop:
                        properties.append(prop)
                except Exception as e:
                    logger.debug(f"Error parsing Zillow card: {e}")
                    continue
            
            logger.info(f"Found {len(properties)} properties on Zillow")
            
            # Cache results
            if properties:
                self.cache.set(address, "zillow", properties)
            
        except Exception as e:
            logger.error(f"Error searching Zillow: {str(e)}")
            self.rate_limiter.record_failure("zillow", error=str(e))
        
        return properties
    
    def _parse_zillow_card(self, card, original_address: str) -> Optional[PropertyDetails]:
        """Parse a Zillow property card to extract details."""
        try:
            # Extract address
            address_elem = card.select_one('address, [data-test="property-card-addr"], h3')
            if not address_elem:
                return None
            
            address = address_elem.get_text(strip=True)
            
            # Extract listing URL
            link_elem = card.select_one('a[href*="/homedetails/"]')
            listing_url = None
            if link_elem:
                listing_url = urljoin("https://www.zillow.com", link_elem.get('href', ''))
            
            # Extract price
            price_elem = card.select_one('[data-test="property-card-price"], span[class*="price"]')
            price = price_elem.get_text(strip=True) if price_elem else None
            
            # Extract property details
            details_elem = card.select_one('[data-test="property-card-details"], ul[class*="StyledPropertyCardHomeDetails"]')
            
            bedrooms = None
            bathrooms = None
            square_feet = None
            
            if details_elem:
                details_text = details_elem.get_text()
                
                # Extract bedrooms
                bed_match = re.search(r'(\d+)\s*(?:bd|bed|bedroom)', details_text, re.IGNORECASE)
                if bed_match:
                    bedrooms = int(bed_match.group(1))
                
                # Extract bathrooms
                bath_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ba|bath|bathroom)', details_text, re.IGNORECASE)
                if bath_match:
                    bathrooms = float(bath_match.group(1))
                
                # Extract square feet
                sqft_match = re.search(r'(\d+(?:,\d+)?)\s*(?:sqft|sq\s*ft|square)', details_text, re.IGNORECASE)
                if sqft_match:
                    square_feet = int(sqft_match.group(1).replace(',', ''))
            
            # Calculate confidence score
            confidence = self._calculate_address_similarity(original_address, address)
            
            return PropertyDetails(
                address=address,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                square_feet=square_feet,
                price=price,
                listing_url=listing_url,
                source="Zillow",
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Zillow card: {e}")
            return None
    
    def _extract_zillow_from_json(self, data: Dict, original_address: str) -> List[PropertyDetails]:
        """Extract property data from Zillow's JSON response."""
        properties = []
        
        def search_for_listings(obj, depth=0):
            """Recursively search for listing data in JSON."""
            if depth > 10:  # Prevent infinite recursion
                return
            
            if isinstance(obj, dict):
                # Check if this looks like a property listing
                if 'address' in obj or 'streetAddress' in obj:
                    prop = self._parse_zillow_json_listing(obj, original_address)
                    if prop:
                        properties.append(prop)
                
                # Continue searching
                for value in obj.values():
                    search_for_listings(value, depth + 1)
                    
            elif isinstance(obj, list):
                for item in obj:
                    search_for_listings(item, depth + 1)
        
        search_for_listings(data)
        return properties
    
    def _parse_zillow_json_listing(self, listing: Dict, original_address: str) -> Optional[PropertyDetails]:
        """Parse a single listing from Zillow JSON data."""
        try:
            # Extract address
            address = None
            if 'address' in listing:
                if isinstance(listing['address'], str):
                    address = listing['address']
                elif isinstance(listing['address'], dict):
                    parts = []
                    for key in ['streetAddress', 'city', 'state', 'zipcode']:
                        if key in listing['address']:
                            parts.append(str(listing['address'][key]))
                    address = ', '.join(parts)
            
            if not address:
                return None
            
            # Extract other details
            bedrooms = listing.get('beds') or listing.get('bedrooms')
            bathrooms = listing.get('baths') or listing.get('bathrooms')
            square_feet = listing.get('area') or listing.get('livingArea')
            price = listing.get('price') or listing.get('priceLabel')
            
            # Calculate confidence
            confidence = self._calculate_address_similarity(original_address, address)
            
            return PropertyDetails(
                address=address,
                bedrooms=int(bedrooms) if bedrooms else None,
                bathrooms=float(bathrooms) if bathrooms else None,
                square_feet=int(square_feet) if square_feet else None,
                price=str(price) if price else None,
                source="Zillow",
                confidence_score=confidence,
                raw_data=listing
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Zillow JSON listing: {e}")
            return None
    
    def search_realtor(self, address: str, features: Optional[Dict] = None) -> List[PropertyDetails]:
        """
        Search Realtor.com for properties matching the given address.
        
        Args:
            address: Property address to search for
            features: Optional dict with property features for verification
            
        Returns:
            List of PropertyDetails objects for matching properties
        """
        # Check cache first
        cached_results = self.cache.get(address, "realtor")
        if cached_results is not None:
            return cached_results
        
        properties = []
        
        try:
            # Check if service is available
            can_request, reason = self.rate_limiter.can_make_request("realtor")
            if not can_request:
                logger.warning(f"Realtor.com search skipped: {reason}")
                return properties
            
            # Format address for Realtor.com search
            search_query = quote_plus(address)
            search_url = f"https://www.realtor.com/realestateandhomes-search/{search_query}"
            
            logger.info(f"Searching Realtor.com for: {address}")
            
            response = self._make_request(search_url, "realtor")
            if not response:
                logger.warning("Failed to get Realtor.com search results")
                return properties
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple selectors for Realtor.com
            selectors = [
                'div[data-testid="card-container"]',
                'li[data-testid="result-card"]',
                'div[class*="CardContent"]',
                'section[class*="property-card"]'
            ]
            
            property_cards = []
            for selector in selectors:
                property_cards = soup.select(selector)
                if property_cards:
                    break
            
            if not property_cards:
                # Try to find properties in Next.js data
                next_data = soup.find('script', id='__NEXT_DATA__')
                if next_data:
                    try:
                        data = json.loads(next_data.string)
                        properties_from_json = self._extract_realtor_from_json(data, address)
                        if properties_from_json:
                            properties.extend(properties_from_json)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Error parsing Realtor.com JSON: {e}")
            
            # Process property cards from HTML
            for card in property_cards[:10]:  # Limit to first 10 results
                try:
                    prop = self._parse_realtor_card(card, address)
                    if prop:
                        properties.append(prop)
                except Exception as e:
                    logger.debug(f"Error parsing Realtor.com card: {e}")
                    continue
            
            logger.info(f"Found {len(properties)} properties on Realtor.com")
            
            # Cache results
            if properties:
                self.cache.set(address, "realtor", properties)
            
        except Exception as e:
            logger.error(f"Error searching Realtor.com: {str(e)}")
            self.rate_limiter.record_failure("realtor", error=str(e))
        
        return properties
    
    def _parse_realtor_card(self, card, original_address: str) -> Optional[PropertyDetails]:
        """Parse a Realtor.com property card to extract details."""
        try:
            # Extract address - Realtor.com often splits it
            address_line = card.select_one('[data-testid="card-address-1"], [class*="address-line"]')
            address_city = card.select_one('[data-testid="card-address-2"], [class*="address-city"]')
            
            if not address_line:
                return None
            
            address_parts = [address_line.get_text(strip=True)]
            if address_city:
                address_parts.append(address_city.get_text(strip=True))
            
            address = ', '.join(address_parts)
            
            # Extract price
            price_elem = card.select_one('[data-testid="card-price"], span[class*="price"]')
            price = price_elem.get_text(strip=True) if price_elem else None
            
            # Extract property meta (beds, baths, sqft)
            meta_elem = card.select_one('[data-testid="card-meta"], ul[class*="property-meta"]')
            
            bedrooms = None
            bathrooms = None
            square_feet = None
            
            if meta_elem:
                # Look for bed/bath/sqft spans
                bed_elem = meta_elem.select_one('[data-testid="meta-beds"], span[data-label*="bed"]')
                if bed_elem:
                    bedrooms = self._extract_number(bed_elem.get_text())
                
                bath_elem = meta_elem.select_one('[data-testid="meta-baths"], span[data-label*="bath"]')
                if bath_elem:
                    bathrooms = self._extract_number(bath_elem.get_text())
                
                sqft_elem = meta_elem.select_one('[data-testid="meta-sqft"], span[data-label*="sqft"]')
                if sqft_elem:
                    sqft_text = sqft_elem.get_text()
                    sqft_match = re.search(r'(\d+(?:,\d+)?)', sqft_text)
                    if sqft_match:
                        square_feet = int(sqft_match.group(1).replace(',', ''))
            
            # Extract listing URL
            link_elem = card.select_one('a[href*="/realestateandhomes-detail/"]')
            listing_url = None
            if link_elem:
                listing_url = urljoin("https://www.realtor.com", link_elem.get('href', ''))
            
            # Calculate confidence score
            confidence = self._calculate_address_similarity(original_address, address)
            
            return PropertyDetails(
                address=address,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                square_feet=square_feet,
                price=price,
                listing_url=listing_url,
                source="Realtor.com",
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Realtor.com card: {e}")
            return None
    
    def _extract_realtor_from_json(self, data: Dict, original_address: str) -> List[PropertyDetails]:
        """Extract property data from Realtor.com's Next.js JSON."""
        properties = []
        
        def find_properties(obj, depth=0):
            """Recursively search for property data."""
            if depth > 10:
                return
            
            if isinstance(obj, dict):
                # Check for property-like structure
                if 'location' in obj and 'address' in obj.get('location', {}):
                    prop = self._parse_realtor_json_property(obj, original_address)
                    if prop:
                        properties.append(prop)
                elif 'address' in obj and isinstance(obj['address'], dict):
                    prop = self._parse_realtor_json_property(obj, original_address)
                    if prop:
                        properties.append(prop)
                
                # Continue searching
                for value in obj.values():
                    find_properties(value, depth + 1)
                    
            elif isinstance(obj, list):
                for item in obj:
                    find_properties(item, depth + 1)
        
        find_properties(data)
        return properties
    
    def _parse_realtor_json_property(self, prop_data: Dict, original_address: str) -> Optional[PropertyDetails]:
        """Parse a property from Realtor.com JSON data."""
        try:
            # Extract address
            address = None
            if 'location' in prop_data and 'address' in prop_data['location']:
                addr = prop_data['location']['address']
                parts = []
                for key in ['line', 'street_name', 'street_number', 'city', 'state_code', 'postal_code']:
                    if key in addr and addr[key]:
                        parts.append(str(addr[key]))
                address = ', '.join(parts)
            elif 'address' in prop_data:
                if isinstance(prop_data['address'], str):
                    address = prop_data['address']
                elif isinstance(prop_data['address'], dict):
                    addr = prop_data['address']
                    parts = []
                    for key in ['line', 'city', 'state_code', 'postal_code']:
                        if key in addr and addr[key]:
                            parts.append(str(addr[key]))
                    address = ', '.join(parts)
            
            if not address:
                return None
            
            # Extract details
            description = prop_data.get('description', {})
            bedrooms = description.get('beds')
            bathrooms = description.get('baths') or description.get('baths_full')
            square_feet = description.get('sqft')
            year_built = description.get('year_built')
            lot_size = description.get('lot_sqft')
            property_type = description.get('type')
            
            # Extract price
            price = None
            if 'list_price' in prop_data:
                price = f"${prop_data['list_price']:,}" if isinstance(prop_data['list_price'], (int, float)) else str(prop_data['list_price'])
            
            # Calculate confidence
            confidence = self._calculate_address_similarity(original_address, address)
            
            return PropertyDetails(
                address=address,
                bedrooms=int(bedrooms) if bedrooms else None,
                bathrooms=float(bathrooms) if bathrooms else None,
                square_feet=int(square_feet) if square_feet else None,
                year_built=int(year_built) if year_built else None,
                price=price,
                property_type=property_type,
                lot_size=str(lot_size) if lot_size else None,
                source="Realtor.com",
                confidence_score=confidence,
                raw_data=prop_data
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Realtor.com JSON property: {e}")
            return None
    
    def cross_reference_property(
        self,
        address: str,
        coordinates: Optional[Tuple[float, float]] = None,
        visual_features: Optional[Dict] = None,
        nlp_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main function to cross-reference property across Zillow and Realtor.com.
        Falls back to vision/NLP data when services are unavailable.
        
        Args:
            address: Property address to search for
            coordinates: Optional (latitude, longitude) tuple
            visual_features: Optional dict with visual features from property analysis
            nlp_data: Optional dict with NLP extracted data
            
        Returns:
            Dict containing consolidated results with confidence scores
        """
        logger.info(f"Starting cross-reference for: {address}")
        
        # Get service status
        service_status = self.rate_limiter.get_status()
        
        results = {
            "query_address": address,
            "coordinates": coordinates,
            "timestamp": datetime.now().isoformat(),
            "zillow_results": [],
            "realtor_results": [],
            "best_matches": [],
            "confidence_analysis": {},
            "summary": {},
            "service_status": service_status,
            "fallback_used": False
        }
        
        try:
            # Check if both services are unavailable
            zillow_available = service_status['zillow']['status'] == 'available'
            realtor_available = service_status['realtor']['status'] == 'available'
            
            if not zillow_available and not realtor_available:
                logger.warning("Both real estate services are unavailable, using fallback data")
                results["fallback_used"] = True
                results["fallback_reason"] = "Both services are rate-limited or circuit broken"
                
                # Use vision and NLP data as fallback
                if visual_features or nlp_data:
                    results["summary"] = {
                        "most_likely_address": address,
                        "confidence": 0.6,  # Lower confidence for fallback
                        "data_source": "vision_and_nlp",
                        "message": "Real estate services unavailable, using visual and NLP analysis",
                        "visual_features": visual_features,
                        "nlp_data": nlp_data
                    }
                else:
                    results["summary"] = {
                        "most_likely_address": address,
                        "confidence": 0.3,
                        "message": "Real estate services unavailable, limited data available"
                    }
                
                return results
            
            # Search Zillow (with caching and rate limiting)
            if zillow_available:
                logger.info("Searching Zillow...")
                zillow_properties = self.search_zillow(address, visual_features)
                results["zillow_results"] = [
                    {
                        "address": p.address,
                        "bedrooms": p.bedrooms,
                        "bathrooms": p.bathrooms,
                        "square_feet": p.square_feet,
                        "price": p.price,
                        "confidence": p.confidence_score,
                        "url": p.listing_url,
                        "from_cache": p.from_cache
                    }
                    for p in zillow_properties
                ]
            else:
                logger.info(f"Skipping Zillow: {service_status['zillow']['status']}")
                results["zillow_skipped"] = True
            
            # Search Realtor.com (with caching and rate limiting)
            if realtor_available:
                logger.info("Searching Realtor.com...")
                realtor_properties = self.search_realtor(address, visual_features)
                results["realtor_results"] = [
                    {
                        "address": p.address,
                        "bedrooms": p.bedrooms,
                        "bathrooms": p.bathrooms,
                        "square_feet": p.square_feet,
                        "price": p.price,
                        "year_built": p.year_built,
                        "confidence": p.confidence_score,
                        "url": p.listing_url,
                        "from_cache": p.from_cache
                    }
                    for p in realtor_properties
                ]
            else:
                logger.info(f"Skipping Realtor.com: {service_status['realtor']['status']}")
                results["realtor_skipped"] = True
            
            # Find best matches across available sources
            all_properties = []
            if zillow_available:
                all_properties.extend(zillow_properties if 'zillow_properties' in locals() else [])
            if realtor_available:
                all_properties.extend(realtor_properties if 'realtor_properties' in locals() else [])
            
            # Sort by confidence score
            all_properties.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Group similar addresses
            grouped_matches = self._group_similar_properties(all_properties)
            
            # Process best matches
            for group in grouped_matches[:5]:  # Top 5 unique addresses
                match_data = {
                    "address": group[0].address,
                    "sources": list(set(p.source for p in group)),
                    "confidence": max(p.confidence_score for p in group),
                    "details": {}
                }
                
                # Merge details from all sources
                for prop in group:
                    if prop.bedrooms:
                        match_data["details"]["bedrooms"] = prop.bedrooms
                    if prop.bathrooms:
                        match_data["details"]["bathrooms"] = prop.bathrooms
                    if prop.square_feet:
                        match_data["details"]["square_feet"] = prop.square_feet
                    if prop.year_built:
                        match_data["details"]["year_built"] = prop.year_built
                    if prop.price:
                        match_data["details"]["price"] = prop.price
                    if prop.from_cache:
                        match_data["from_cache"] = True
                
                # Check feature matching if visual features provided
                if visual_features:
                    feature_score = self._calculate_feature_match(match_data["details"], visual_features)
                    match_data["feature_match_score"] = feature_score
                    match_data["confidence"] = (match_data["confidence"] + feature_score) / 2
                
                results["best_matches"].append(match_data)
            
            # Confidence analysis
            results["confidence_analysis"] = {
                "highest_confidence": max([p.confidence_score for p in all_properties]) if all_properties else 0,
                "average_confidence": sum([p.confidence_score for p in all_properties]) / len(all_properties) if all_properties else 0,
                "total_properties_found": len(all_properties),
                "unique_addresses_found": len(grouped_matches)
            }
            
            # Summary
            if results["best_matches"]:
                best = results["best_matches"][0]
                results["summary"] = {
                    "most_likely_address": best["address"],
                    "confidence": best["confidence"],
                    "verified_on": best["sources"],
                    "property_details": best["details"],
                    "from_cache": best.get("from_cache", False)
                }
            else:
                # Fallback to vision/NLP data if no matches found
                if visual_features or nlp_data:
                    results["summary"] = {
                        "most_likely_address": address,
                        "confidence": 0.5,
                        "message": "No real estate matches found, using visual/NLP analysis",
                        "visual_features": visual_features,
                        "nlp_data": nlp_data
                    }
                    results["fallback_used"] = True
                else:
                    results["summary"] = {
                        "most_likely_address": None,
                        "confidence": 0,
                        "message": "No matching properties found"
                    }
            
        except Exception as e:
            logger.error(f"Error in cross-reference: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _group_similar_properties(self, properties: List[PropertyDetails]) -> List[List[PropertyDetails]]:
        """Group properties with similar addresses."""
        if not properties:
            return []
        
        groups = []
        used = set()
        
        for i, prop1 in enumerate(properties):
            if i in used:
                continue
            
            group = [prop1]
            used.add(i)
            
            for j, prop2 in enumerate(properties[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if addresses are similar enough to be the same property
                similarity = self._calculate_address_similarity(prop1.address, prop2.address)
                if similarity > 0.85:  # High threshold for grouping
                    group.append(prop2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_feature_match(self, property_details: Dict, visual_features: Dict) -> float:
        """Calculate how well property details match visual features."""
        if not visual_features:
            return 0.5  # Neutral score if no features to compare
        
        score = 0
        comparisons = 0
        
        # Compare bedrooms
        if "bedrooms" in property_details and "bedrooms" in visual_features:
            if property_details["bedrooms"] == visual_features["bedrooms"]:
                score += 1
            comparisons += 1
        
        # Compare bathrooms
        if "bathrooms" in property_details and "bathrooms" in visual_features:
            if abs(property_details["bathrooms"] - visual_features["bathrooms"]) <= 0.5:
                score += 1
            comparisons += 1
        
        # Compare property type if available
        if "property_type" in property_details and "property_type" in visual_features:
            prop_type1 = property_details["property_type"].lower()
            prop_type2 = visual_features["property_type"].lower()
            if prop_type1 in prop_type2 or prop_type2 in prop_type1:
                score += 1
            comparisons += 1
        
        # Return normalized score
        return (score / comparisons) if comparisons > 0 else 0.5


# Convenience functions for direct use
def search_zillow(address: str, features: Optional[Dict] = None) -> List[PropertyDetails]:
    """
    Search Zillow for properties matching the given address.
    
    Args:
        address: Property address to search for
        features: Optional dict with property features for verification
        
    Returns:
        List of PropertyDetails objects for matching properties
    """
    searcher = RealEstateSearcher()
    return searcher.search_zillow(address, features)


def search_realtor(address: str, features: Optional[Dict] = None) -> List[PropertyDetails]:
    """
    Search Realtor.com for properties matching the given address.
    
    Args:
        address: Property address to search for
        features: Optional dict with property features for verification
        
    Returns:
        List of PropertyDetails objects for matching properties
    """
    searcher = RealEstateSearcher()
    return searcher.search_realtor(address, features)


def search_parallel(address: str, features: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Search both Zillow and Realtor.com in parallel with timeouts.
    Returns partial results if one service fails.
    
    Args:
        address: Property address to search for
        features: Optional dict with property features for verification
        
    Returns:
        Dict containing results from both platforms
    """
    searcher = RealEstateSearcher()
    results = {
        "zillow_matches": [],
        "realtor_matches": [],
        "errors": [],
        "service_status": searcher.rate_limiter.get_status()
    }
    
    def search_zillow_wrapper():
        try:
            logger.info("Starting Zillow search (parallel)")
            return searcher.search_zillow(address, features)
        except Exception as e:
            logger.error(f"Zillow search failed: {e}")
            results["errors"].append(f"Zillow: {str(e)}")
            return []
    
    def search_realtor_wrapper():
        try:
            logger.info("Starting Realtor.com search (parallel)")
            return searcher.search_realtor(address, features)
        except Exception as e:
            logger.error(f"Realtor search failed: {e}")
            results["errors"].append(f"Realtor: {str(e)}")
            return []
    
    # Check service availability before executing
    zillow_available, _ = searcher.rate_limiter.can_make_request("zillow")
    realtor_available, _ = searcher.rate_limiter.can_make_request("realtor")
    
    if not zillow_available and not realtor_available:
        logger.warning("Both services unavailable, returning empty results")
        results["errors"].append("Both services are rate-limited or circuit broken")
        return results
    
    # Execute searches in parallel with timeout (only for available services)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        if zillow_available:
            futures['zillow'] = executor.submit(search_zillow_wrapper)
        else:
            results["errors"].append("Zillow: Service unavailable")
        
        if realtor_available:
            futures['realtor'] = executor.submit(search_realtor_wrapper)
        else:
            results["errors"].append("Realtor: Service unavailable")
        
        # Get results with timeout
        if 'zillow' in futures:
            try:
                results["zillow_matches"] = futures['zillow'].result(timeout=SEARCH_TIMEOUT)
            except TimeoutError:
                logger.warning("Zillow search timed out")
                results["errors"].append("Zillow search timed out")
        
        if 'realtor' in futures:
            try:
                results["realtor_matches"] = futures['realtor'].result(timeout=SEARCH_TIMEOUT)
            except TimeoutError:
                logger.warning("Realtor search timed out")
                results["errors"].append("Realtor search timed out")
    
    logger.info(f"Parallel search complete - Zillow: {len(results['zillow_matches'])} matches, "
                f"Realtor: {len(results['realtor_matches'])} matches")
    
    return results

def cross_reference_property(
    address: str,
    coordinates: Optional[Tuple[float, float]] = None,
    visual_features: Optional[Dict] = None,
    nlp_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Main function to cross-reference property across Zillow and Realtor.com.
    Uses parallel execution for faster results and falls back to vision/NLP data.
    
    Args:
        address: Property address to search for
        coordinates: Optional (latitude, longitude) tuple
        visual_features: Optional dict with visual features from property analysis
        nlp_data: Optional dict with NLP extracted data
        
    Returns:
        Dict containing consolidated results with confidence scores
    """
    searcher = RealEstateSearcher()
    return searcher.cross_reference_property(address, coordinates, visual_features, nlp_data)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample address
    test_address = "1234 Main Street, San Francisco, CA 94102"
    test_coords = (37.7749, -122.4194)
    test_features = {
        "bedrooms": 3,
        "bathrooms": 2,
        "property_type": "single family home"
    }
    
    print("Testing Enhanced Real Estate Searcher Module")
    print("=" * 50)
    
    # Create searcher to check service status
    searcher = RealEstateSearcher()
    status = searcher.rate_limiter.get_status()
    print("\nService Status:")
    for service, state in status.items():
        print(f"  {service}: {state['status']} (failures: {state['consecutive_failures']})")
    
    print("\n1. Testing cross-reference with fallback:")
    results = cross_reference_property(test_address, test_coords, test_features)
    
    if results.get("summary"):
        summary = results["summary"]
        print(f"Most likely address: {summary.get('most_likely_address')}")
        print(f"Confidence: {summary.get('confidence', 0):.2%}")
        if summary.get('from_cache'):
            print("Results from cache")
        if results.get('fallback_used'):
            print(f"Fallback used: {results.get('fallback_reason')}")
        print(f"Data source: {summary.get('data_source', 'real_estate_sites')}")
        
        if summary.get("property_details"):
            print("Property details:")
            for key, value in summary["property_details"].items():
                print(f"  - {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")