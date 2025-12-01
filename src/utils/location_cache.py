"""
Location data cache module with thread-safe operations and TTL management.
This module provides caching for Airbnb location data to avoid repeated API calls.
"""

import time
import logging
import hashlib
import json
import threading
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

class LocationDataCache:
    """Thread-safe in-memory cache for Airbnb location data with TTL support."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._total_requests = 0
        
        # Start background cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"LocationDataCache initialized with TTL={ttl_seconds}s")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from the URL."""
        # Normalize URL and create hash
        normalized = url.lower().strip().rstrip('/')
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached location data for a URL.
        
        Args:
            url: The Airbnb listing URL
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        with self._lock:
            self._total_requests += 1
            cache_key = self._get_cache_key(url)
            
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                
                # Check if cache entry has expired
                if time.time() - timestamp < self.ttl_seconds:
                    self._hit_count += 1
                    hit_rate = (self._hit_count / self._total_requests) * 100
                    logger.info(f"Cache HIT for URL: {url[:50]}... (hit rate: {hit_rate:.1f}%)")
                    
                    # Update Prometheus metrics
                    if METRICS_AVAILABLE:
                        cache_hits.labels(cache_type='location_data').inc()
                        cache_operations.labels(cache_type='location_data', operation='get_hit').inc()
                    
                    # Return deep copy to prevent modification
                    return json.loads(json.dumps(data))
                else:
                    # Expired entry - remove it
                    del self._cache[cache_key]
                    logger.debug(f"Cache entry expired for URL: {url[:50]}...")
                    if METRICS_AVAILABLE:
                        cache_operations.labels(cache_type='location_data', operation='evict_expired').inc()
            
            self._miss_count += 1
            hit_rate = (self._hit_count / self._total_requests) * 100 if self._total_requests > 0 else 0
            logger.info(f"Cache MISS for URL: {url[:50]}... (hit rate: {hit_rate:.1f}%)")
            
            # Update Prometheus metrics
            if METRICS_AVAILABLE:
                cache_misses.labels(cache_type='location_data').inc()
                cache_operations.labels(cache_type='location_data', operation='get_miss').inc()
                
            return None
    
    def set(self, url: str, data: Dict[str, Any]) -> None:
        """
        Store location data in cache.
        
        Args:
            url: The Airbnb listing URL
            data: Location data to cache
        """
        with self._lock:
            cache_key = self._get_cache_key(url)
            # Store deep copy to prevent external modifications
            data_copy = json.loads(json.dumps(data, default=str))
            self._cache[cache_key] = (data_copy, time.time())
            logger.debug(f"Cached data for URL: {url[:50]}... (cache size: {len(self._cache)})")
            
            # Update Prometheus metrics
            if METRICS_AVAILABLE:
                cache_operations.labels(cache_type='location_data', operation='set').inc()
                cache_size.labels(cache_type='location_data').set(len(self._cache))
            
            # Clean up if cache gets too large
            if len(self._cache) > 500:
                self._evict_oldest(100)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cache entries")
    
    def _evict_oldest(self, count: int) -> None:
        """
        Evict the oldest cache entries.
        
        Args:
            count: Number of entries to evict
        """
        # Sort by timestamp and remove oldest
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_entries[:count]:
            del self._cache[key]
        logger.debug(f"Evicted {count} oldest cache entries")
    
    def _cleanup_expired(self) -> None:
        """Background thread to clean up expired entries periodically."""
        while self._running:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                with self._lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, (_, timestamp) in self._cache.items()
                        if current_time - timestamp >= self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self._hit_count / self._total_requests * 100) if self._total_requests > 0 else 0
            return {
                "total_requests": self._total_requests,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "ttl_seconds": self.ttl_seconds
            }
    
    def shutdown(self) -> None:
        """Shutdown the cache and cleanup thread."""
        self._running = False
        logger.info("LocationDataCache shutdown initiated")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Global cache instance
_location_cache = LocationDataCache(ttl_seconds=3600)


def get_cache() -> LocationDataCache:
    """Get the global cache instance."""
    return _location_cache


def cached_location_data(func):
    """
    Decorator to add caching to location data extraction functions.
    
    Usage:
        @cached_location_data
        def get_airbnb_location_data(url):
            # ... expensive operation ...
            return data
    """
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        # Check if caching is disabled
        if kwargs.get('skip_cache', False):
            kwargs.pop('skip_cache', None)
            return func(url, *args, **kwargs)
        
        # Try to get from cache first
        cache = get_cache()
        cached_data = cache.get(url)
        
        if cached_data is not None:
            # Add metadata to indicate this came from cache
            cached_data['_from_cache'] = True
            cached_data['_cache_timestamp'] = time.time()
            return cached_data
        
        # Not in cache - call the function
        result = func(url, *args, **kwargs)
        
        # Cache the result if successful
        if result and not result.get('error'):
            cache.set(url, result)
            result['_from_cache'] = False
        
        return result
    
    return wrapper


# Integration with Prometheus metrics
try:
    from core.metrics import (
        cache_hits, cache_misses, cache_operations,
        cache_size, cache_hit_rate
    )
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Prometheus metrics not available for cache monitoring")
    METRICS_AVAILABLE = False


def update_cache_metrics():
    """Update Prometheus metrics with cache statistics."""
    if not METRICS_AVAILABLE:
        return
    
    stats = _location_cache.get_stats()
    
    # Update cache metrics
    cache_size.labels(cache_type='location_data').set(stats['cache_size'])
    cache_hit_rate.labels(cache_type='location_data').set(stats['hit_rate'])
    
    logger.debug(f"Cache metrics updated: hit_rate={stats['hit_rate']:.1f}%, size={stats['cache_size']}")