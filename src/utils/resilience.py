"""
Resilience Patterns Module
Provides circuit breaker, retry mechanisms, and fallback strategies for external API calls.
"""

import os
import logging
import functools
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta

from circuitbreaker import circuit
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log,
    RetryError
)

logger = logging.getLogger(__name__)

# Configuration settings (can be overridden via environment variables)
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.environ.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5))
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.environ.get('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', 60))
CIRCUIT_BREAKER_EXPECTED_EXCEPTION = Exception

MAX_RETRY_ATTEMPTS = int(os.environ.get('MAX_RETRY_ATTEMPTS', 3))
MIN_RETRY_WAIT = int(os.environ.get('MIN_RETRY_WAIT', 2))
MAX_RETRY_WAIT = int(os.environ.get('MAX_RETRY_WAIT', 10))

# Circuit breaker configurations for different services
CIRCUIT_CONFIGS = {
    'openai': {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'expected_exception': Exception,
        'name': 'openai_circuit'
    },
    'google_vision': {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'expected_exception': Exception,
        'name': 'google_vision_circuit'
    },
    'apify': {
        'failure_threshold': 3,
        'recovery_timeout': 120,
        'expected_exception': Exception,
        'name': 'apify_circuit'
    },
    'google_maps': {
        'failure_threshold': 5,
        'recovery_timeout': 30,
        'expected_exception': Exception,
        'name': 'google_maps_circuit'
    }
}

# Store circuit breaker instances
circuit_breakers = {}


def get_circuit_breaker(service_name: str) -> Callable:
    """
    Get or create a circuit breaker for a specific service.
    
    Args:
        service_name: Name of the service (e.g., 'openai', 'google_vision')
        
    Returns:
        Circuit breaker decorator for the service
    """
    if service_name not in circuit_breakers:
        config = CIRCUIT_CONFIGS.get(service_name, {
            'failure_threshold': CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            'recovery_timeout': CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            'expected_exception': CIRCUIT_BREAKER_EXPECTED_EXCEPTION,
            'name': f'{service_name}_circuit'
        })
        
        circuit_breakers[service_name] = circuit(
            failure_threshold=config['failure_threshold'],
            recovery_timeout=config['recovery_timeout'],
            expected_exception=config['expected_exception'],
            name=config['name']
        )
    
    return circuit_breakers[service_name]


def resilient_call(
    service_name: str,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    min_wait: int = MIN_RETRY_WAIT,
    max_wait: int = MAX_RETRY_WAIT,
    fallback: Optional[Callable] = None,
    retry_exceptions: tuple = (Exception,)
):
    """
    Decorator that combines circuit breaker and retry patterns.
    
    Args:
        service_name: Name of the service for circuit breaker
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        fallback: Optional fallback function to call if all retries fail
        retry_exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function with resilience patterns
    """
    def decorator(func: Callable) -> Callable:
        # Apply retry pattern
        retry_decorator = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_exceptions),
            before=before_log(logger, logging.DEBUG),
            after=after_log(logger, logging.DEBUG),
            reraise=True
        )
        
        # Apply circuit breaker
        circuit_decorator = get_circuit_breaker(service_name)
        
        # Create the resilient function ONCE at decoration time
        # This ensures the circuit breaker state is preserved across calls
        @circuit_decorator
        @retry_decorator
        @functools.wraps(func)
        def resilient_func(*args, **kwargs):
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Call the pre-created resilient function
                return resilient_func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"All retry attempts failed for {service_name}: {str(e)}")
                
                # If we have a fallback, use it
                if fallback:
                    logger.info(f"Using fallback for {service_name}")
                    return fallback(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator


class ResilienceManager:
    """
    Manager class for tracking service health and implementing fallback strategies.
    """
    
    def __init__(self):
        self.service_health: Dict[str, Dict] = {}
        self.fallback_registry: Dict[str, Callable] = {}
        
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """
        Register a fallback function for a service.
        
        Args:
            service_name: Name of the service
            fallback_func: Function to call when service fails
        """
        self.fallback_registry[service_name] = fallback_func
        logger.info(f"Registered fallback for {service_name}")
        
    def get_fallback(self, service_name: str) -> Optional[Callable]:
        """
        Get the fallback function for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Fallback function or None if not registered
        """
        return self.fallback_registry.get(service_name)
    
    def update_service_health(self, service_name: str, is_healthy: bool, error: Optional[str] = None):
        """
        Update the health status of a service.
        
        Args:
            service_name: Name of the service
            is_healthy: Whether the service is healthy
            error: Optional error message
        """
        if service_name not in self.service_health:
            self.service_health[service_name] = {
                'is_healthy': True,
                'failures': 0,
                'last_failure': None,
                'last_success': None,
                'last_error': None
            }
        
        health = self.service_health[service_name]
        
        if is_healthy:
            health['is_healthy'] = True
            health['failures'] = 0
            health['last_success'] = datetime.now()
        else:
            health['is_healthy'] = False
            health['failures'] += 1
            health['last_failure'] = datetime.now()
            health['last_error'] = error
            
        logger.debug(f"Service {service_name} health updated: {health}")
    
    def is_service_healthy(self, service_name: str) -> bool:
        """
        Check if a service is healthy.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is healthy, False otherwise
        """
        if service_name not in self.service_health:
            return True  # Assume healthy if no history
        
        return self.service_health[service_name]['is_healthy']
    
    def get_service_status(self) -> Dict[str, Dict]:
        """
        Get the current status of all services.
        
        Returns:
            Dictionary of service statuses
        """
        return self.service_health.copy()


# Global resilience manager instance
resilience_manager = ResilienceManager()


# Fallback functions for specific services
def openai_fallback(*args, **kwargs) -> Dict[str, Any]:
    """
    Fallback for OpenAI API calls - uses rule-based processing.
    """
    logger.warning("Using rule-based fallback for OpenAI API")
    
    # Extract the type of analysis from kwargs or args
    # This is a simplified fallback that returns structured defaults
    return {
        "key_location_hints": [],
        "distance_mentions": [],
        "landmark_mentions": [],
        "neighborhood_insights": "OpenAI API unavailable - using fallback response",
        "fallback_used": True
    }


def google_vision_fallback(image_bytes: bytes) -> Dict[str, Any]:
    """
    Fallback for Google Vision API - uses Tesseract OCR.
    """
    logger.warning("Using Tesseract OCR fallback for Google Vision")
    
    try:
        # Import Tesseract OCR module
        from ocr.tesseract_ocr import TesseractHouseNumberDetector
        
        # Use Tesseract as fallback
        tesseract_ocr = TesseractHouseNumberDetector()
        
        # Convert bytes to numpy array for Tesseract
        import io
        from PIL import Image
        import numpy as np
        import cv2
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        result = tesseract_ocr.detect_house_numbers(img_array)
        result['fallback_used'] = True
        result['fallback_method'] = 'tesseract_ocr'
        
        return result
        
    except Exception as e:
        logger.error(f"Tesseract fallback also failed: {str(e)}")
        return {
            'error': f'Both Google Vision and Tesseract failed: {str(e)}',
            'house_numbers_found': [],
            'all_text': [],
            'fallback_used': True,
            'fallback_method': 'tesseract_ocr_failed'
        }


def apify_fallback(listing_url: str) -> Dict[str, Any]:
    """
    Fallback for Apify scraper - uses basic scraper.
    """
    logger.warning("Using basic scraper fallback for Apify")
    
    try:
        # Import basic scraper module
        from extraction.scraper import get_airbnb_location_data
        
        # Use basic scraper as fallback
        result = get_airbnb_location_data(listing_url)
        if result:
            result['fallback_used'] = True
            result['fallback_method'] = 'basic_scraper'
            result['apify_success'] = False
        else:
            result = {
                'fallback_used': True,
                'fallback_method': 'basic_scraper',
                'apify_success': False,
                'error': 'Basic scraper returned no data'
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Basic scraper fallback also failed: {str(e)}")
        return {
            'error': f'Both Apify and basic scraper failed: {str(e)}',
            'fallback_used': True,
            'fallback_method': 'basic_scraper_failed',
            'apify_success': False
        }


# Register default fallbacks
resilience_manager.register_fallback('openai', openai_fallback)
resilience_manager.register_fallback('google_vision', google_vision_fallback)
resilience_manager.register_fallback('apify', apify_fallback)


# Utility functions for monitoring
def get_circuit_breaker_status(service_name: str) -> Dict[str, Any]:
    """
    Get the current status of a circuit breaker.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Dictionary with circuit breaker status
    """
    if service_name not in circuit_breakers:
        return {'exists': False, 'service': service_name}
    
    cb = circuit_breakers[service_name]
    
    # CircuitBreaker object attributes vary, so handle gracefully
    status = {
        'exists': True,
        'service': service_name,
        'state': 'closed',  # Default state
        'failure_count': 0,
        'last_failure': None,
        'success_count': 0
    }
    
    # Try to get actual state
    if hasattr(cb, 'state'):
        status['state'] = str(cb.state)
    elif hasattr(cb, 'current_state'):
        status['state'] = str(cb.current_state)
    elif hasattr(cb, '_state'):
        status['state'] = str(cb._state)
    
    # Get failure count
    if hasattr(cb, 'failure_count'):
        status['failure_count'] = cb.failure_count
    elif hasattr(cb, '_failure_count'):
        status['failure_count'] = cb._failure_count
        
    # Get last failure time
    if hasattr(cb, 'last_failure_time'):
        status['last_failure'] = cb.last_failure_time
    elif hasattr(cb, '_last_failure'):
        status['last_failure'] = cb._last_failure
        
    # Get success count
    if hasattr(cb, 'success_count'):
        status['success_count'] = cb.success_count
    elif hasattr(cb, '_success_count'):
        status['success_count'] = cb._success_count
        
    return status


def get_all_circuit_breaker_statuses() -> Dict[str, Dict]:
    """
    Get the status of all circuit breakers.
    
    Returns:
        Dictionary of all circuit breaker statuses
    """
    return {
        service: get_circuit_breaker_status(service)
        for service in CIRCUIT_CONFIGS.keys()
    }


# Example usage decorator for OpenAI calls
def resilient_openai_call(func: Callable) -> Callable:
    """
    Decorator specifically for OpenAI API calls with built-in resilience.
    """
    return resilient_call(
        service_name='openai',
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        fallback=openai_fallback,
        retry_exceptions=(Exception,)
    )(func)


# Example usage decorator for Google Vision calls
def resilient_google_vision_call(func: Callable) -> Callable:
    """
    Decorator specifically for Google Vision API calls with built-in resilience.
    """
    return resilient_call(
        service_name='google_vision',
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        fallback=google_vision_fallback,
        retry_exceptions=(Exception,)
    )(func)


# Example usage decorator for Apify calls
def resilient_apify_call(func: Callable) -> Callable:
    """
    Decorator specifically for Apify API calls with built-in resilience.
    """
    return resilient_call(
        service_name='apify',
        max_attempts=2,
        min_wait=3,
        max_wait=15,
        fallback=apify_fallback,
        retry_exceptions=(Exception,)
    )(func)


# Example usage decorator for Google Maps calls
def resilient_google_maps_call(func: Callable) -> Callable:
    """
    Decorator specifically for Google Maps API calls with built-in resilience.
    """
    return resilient_call(
        service_name='google_maps',
        max_attempts=3,
        min_wait=1,
        max_wait=5,
        fallback=None,  # No specific fallback for Google Maps
        retry_exceptions=(Exception,)
    )(func)