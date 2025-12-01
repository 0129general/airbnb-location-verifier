"""
Prometheus metrics definitions for monitoring the Airbnb Address Verification system.
Includes counters, histograms, gauges, and summary metrics for comprehensive observability.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, REGISTRY, CollectorRegistry
)
import time
import psutil
import logging
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# =======================
# VERIFICATION METRICS
# =======================

# Counter for total verification requests
verification_counter = Counter(
    'verification_total',
    'Total number of verification requests',
    ['status', 'mode']  # status: success/failure, mode: quick/thorough
)

# Histogram for verification processing time
verification_time = Histogram(
    'verification_processing_seconds',
    'Time spent processing verifications',
    ['stage'],  # stage: extraction/ocr/scoring/total
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)
)

# Gauge for confidence scores
confidence_gauge = Gauge(
    'verification_confidence_score',
    'Latest confidence score for verifications'
)

# Counter for different address sources
address_source_counter = Counter(
    'address_source_total',
    'Count of addresses by source',
    ['source']  # source: scraping/ocr/visual_analysis/multi_signal
)

# =======================
# API METRICS
# =======================

# Counter for HTTP requests
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

# Histogram for response times
http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latencies',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

# =======================
# BACKGROUND JOB METRICS
# =======================

# Gauge for active background jobs
active_jobs = Gauge(
    'background_jobs_active',
    'Number of active background jobs',
    ['job_type']
)

# Counter for job completions
job_completions = Counter(
    'background_jobs_completed_total',
    'Total completed background jobs',
    ['job_type', 'status']  # status: success/failure
)

# Histogram for job processing times
job_processing_time = Histogram(
    'background_job_duration_seconds',
    'Background job processing duration',
    ['job_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800)
)

# =======================
# EXTERNAL SERVICE METRICS
# =======================

# Counter for external API calls
external_api_calls = Counter(
    'external_api_calls_total',
    'Total external API calls',
    ['service', 'endpoint', 'status']
)

# Histogram for external API response times
external_api_latency = Histogram(
    'external_api_latency_seconds',
    'External API call latencies',
    ['service', 'endpoint'],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30)
)

# Counter for API errors
api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['service', 'error_type']
)

# =======================
# CACHE METRICS
# =======================

# Counter for cache hits
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']  # cache_type: location_data, google_search, etc.
)

# Counter for cache misses
cache_misses = Counter(
    'cache_misses_total', 
    'Total cache misses',
    ['cache_type']
)

# Gauge for cache size
cache_size = Gauge(
    'cache_entries_count',
    'Number of entries in cache',
    ['cache_type']
)

# Gauge for cache hit rate
cache_hit_rate = Gauge(
    'cache_hit_rate_percent',
    'Cache hit rate percentage',
    ['cache_type']
)

# Counter for cache operations
cache_operations = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['cache_type', 'operation']  # operation: get, set, evict, clear
)

# =======================
# VISION/OCR METRICS
# =======================

# Counter for OCR operations
ocr_operations = Counter(
    'ocr_operations_total',
    'Total OCR operations',
    ['provider', 'status']  # provider: google_vision/tesseract, status: success/failure
)

# Histogram for OCR processing time
ocr_processing_time = Histogram(
    'ocr_processing_seconds',
    'OCR processing duration',
    ['provider'],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60)
)

# Gauge for OCR accuracy/confidence
ocr_confidence = Gauge(
    'ocr_confidence_score',
    'OCR confidence scores',
    ['provider']
)

# Counter for images processed
images_processed = Counter(
    'images_processed_total',
    'Total images processed',
    ['processing_type']  # type: ocr/visual_analysis/street_view
)

# =======================
# DATABASE METRICS
# =======================

# Counter for database operations
db_operations = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation', 'table', 'status']
)

# Histogram for query times
db_query_time = Histogram(
    'database_query_seconds',
    'Database query execution time',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5)
)

# Gauge for active database connections
db_connections = Gauge(
    'database_connections_active',
    'Active database connections'
)

# =======================
# SYSTEM METRICS
# =======================

# Gauge for system resources
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
memory_bytes = Gauge('system_memory_bytes', 'Memory usage in bytes')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

# Gauge for application uptime
app_uptime = Gauge('application_uptime_seconds', 'Application uptime in seconds')

# Info metric for application version and metadata
app_info = Info('application_info', 'Application metadata')

# =======================
# SESSION METRICS
# =======================

# Gauge for active sessions
active_sessions = Gauge(
    'active_sessions_total',
    'Number of active verification sessions'
)

# Counter for session cleanup
sessions_cleaned = Counter(
    'sessions_cleaned_total',
    'Total sessions cleaned up'
)

# =======================
# ALERT METRICS
# =======================

# Gauge for alert thresholds
alert_threshold = Gauge(
    'alert_threshold_value',
    'Current alert threshold values',
    ['alert_type', 'metric']
)

# Counter for triggered alerts
alerts_triggered = Counter(
    'alerts_triggered_total',
    'Total alerts triggered',
    ['alert_type', 'severity']
)

# =======================
# HELPER FUNCTIONS & DECORATORS
# =======================

def track_request_metrics(method, endpoint, status_code):
    """Track HTTP request metrics."""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()

def track_verification_metrics(status, mode='thorough', confidence=None):
    """Track verification metrics."""
    verification_counter.labels(status=status, mode=mode).inc()
    if confidence is not None:
        confidence_gauge.set(confidence)

def track_external_api_call(service, endpoint, status, latency=None):
    """Track external API call metrics."""
    external_api_calls.labels(
        service=service,
        endpoint=endpoint,
        status=status
    ).inc()
    
    if latency is not None:
        external_api_latency.labels(
            service=service,
            endpoint=endpoint
        ).observe(latency)

def track_ocr_operation(provider, status, processing_time=None, confidence=None):
    """Track OCR operation metrics."""
    ocr_operations.labels(provider=provider, status=status).inc()
    
    if processing_time is not None:
        ocr_processing_time.labels(provider=provider).observe(processing_time)
    
    if confidence is not None:
        ocr_confidence.labels(provider=provider).set(confidence)

def update_system_metrics():
    """Update system resource metrics."""
    try:
        # CPU usage
        cpu_usage.set(psutil.cpu_percent(interval=1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage.set(memory.percent)
        memory_bytes.set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage.set(disk.percent)
        
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")

def measure_time(histogram, **labels):
    """Decorator to measure execution time of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                histogram.labels(**labels).observe(duration)
        return wrapper
    return decorator

def track_background_job(job_type):
    """Context manager for tracking background job metrics."""
    class JobTracker:
        def __enter__(self):
            self.start_time = time.time()
            self.job_type = job_type
            active_jobs.labels(job_type=job_type).inc()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            active_jobs.labels(job_type=self.job_type).dec()
            duration = time.time() - self.start_time
            job_processing_time.labels(job_type=self.job_type).observe(duration)
            
            status = 'failure' if exc_type else 'success'
            job_completions.labels(job_type=self.job_type, status=status).inc()
            
            if exc_type:
                logger.error(f"Background job {self.job_type} failed: {exc_val}")
            
    return JobTracker()

def track_database_operation(operation, table):
    """Context manager for tracking database operations."""
    class DbTracker:
        def __enter__(self):
            self.start_time = time.time()
            self.operation = operation
            self.table = table
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            status = 'failure' if exc_type else 'success'
            
            db_operations.labels(
                operation=self.operation,
                table=self.table,
                status=status
            ).inc()
            
            db_query_time.labels(
                operation=self.operation,
                table=self.table
            ).observe(duration)
    
    return DbTracker()

# =======================
# ALERT CONFIGURATION
# =======================

class AlertManager:
    """Manages alert thresholds and triggering."""
    
    def __init__(self):
        self.thresholds = {
            'high_error_rate': {
                'metric': 'error_rate',
                'value': 0.1,  # 10% error rate
                'severity': 'warning'
            },
            'slow_response': {
                'metric': 'response_time',
                'value': 5.0,  # 5 seconds
                'severity': 'warning'
            },
            'high_cpu': {
                'metric': 'cpu_usage',
                'value': 80.0,  # 80%
                'severity': 'warning'
            },
            'high_memory': {
                'metric': 'memory_usage',
                'value': 90.0,  # 90%
                'severity': 'critical'
            },
            'job_queue_backlog': {
                'metric': 'job_queue_size',
                'value': 100,  # 100 pending jobs
                'severity': 'warning'
            }
        }
        
        # Initialize threshold gauges
        for alert_type, config in self.thresholds.items():
            alert_threshold.labels(
                alert_type=alert_type,
                metric=config['metric']
            ).set(config['value'])
    
    def check_alert(self, alert_type, current_value):
        """Check if an alert should be triggered."""
        if alert_type not in self.thresholds:
            return False
        
        threshold = self.thresholds[alert_type]
        should_alert = current_value > threshold['value']
        
        if should_alert:
            alerts_triggered.labels(
                alert_type=alert_type,
                severity=threshold['severity']
            ).inc()
            logger.warning(
                f"Alert triggered: {alert_type} - "
                f"Current value: {current_value}, "
                f"Threshold: {threshold['value']}"
            )
        
        return should_alert
    
    def update_threshold(self, alert_type, new_value):
        """Update an alert threshold."""
        if alert_type in self.thresholds:
            self.thresholds[alert_type]['value'] = new_value
            alert_threshold.labels(
                alert_type=alert_type,
                metric=self.thresholds[alert_type]['metric']
            ).set(new_value)
            logger.info(f"Updated threshold for {alert_type} to {new_value}")

# Initialize alert manager
alert_manager = AlertManager()

# =======================
# METRICS COLLECTION
# =======================

def collect_metrics():
    """Collect all metrics and return in Prometheus format."""
    # Update system metrics before collection
    update_system_metrics()
    
    # Return metrics in Prometheus text format
    return generate_latest()

def get_metrics_summary():
    """Get a summary of current metrics for dashboard display."""
    # Update system metrics
    update_system_metrics()
    
    # Helper function to safely get metric value
    def get_counter_total(counter):
        """Safely get total value from a Counter metric."""
        total = 0
        try:
            # Collect all samples from the counter
            for metric in counter.collect():
                for sample in metric.samples:
                    # For counters, we want the _total sample
                    if sample.name.endswith('_total'):
                        total += sample.value
        except Exception:
            pass
        return total
    
    def get_gauge_value(gauge):
        """Safely get current value from a Gauge metric."""
        try:
            # Collect samples from the gauge
            for metric in gauge.collect():
                for sample in metric.samples:
                    # Return the first sample value (gauges typically have one value)
                    return sample.value
        except Exception:
            pass
        return 0
    
    def get_counter_with_labels_total(counter):
        """Safely get total value from a Counter with labels."""
        total = 0
        try:
            # Collect all samples from the counter
            for metric in counter.collect():
                for sample in metric.samples:
                    # Sum all labeled values
                    if sample.name.endswith('_total'):
                        total += sample.value
        except Exception:
            pass
        return total
    
    def get_gauge_with_labels_total(gauge):
        """Safely get total value from a Gauge with labels."""
        total = 0
        try:
            # Collect all samples from the gauge
            for metric in gauge.collect():
                for sample in metric.samples:
                    # Sum all labeled values for gauges (e.g., active jobs by type)
                    total += sample.value
        except Exception:
            pass
        return total
    
    # Collect current metric values using proper prometheus-client API
    summary = {
        'verification': {
            'total_requests': get_counter_with_labels_total(verification_counter),
            'latest_confidence': get_gauge_value(confidence_gauge),
        },
        'system': {
            'cpu_usage': get_gauge_value(cpu_usage),
            'memory_usage': get_gauge_value(memory_usage),
            'memory_bytes': get_gauge_value(memory_bytes),
            'disk_usage': get_gauge_value(disk_usage),
        },
        'background_jobs': {
            'active': get_gauge_with_labels_total(active_jobs),
            'completed': get_counter_with_labels_total(job_completions),
        },
        'database': {
            'total_operations': get_counter_with_labels_total(db_operations),
            'active_connections': get_gauge_value(db_connections),
        },
        'external_apis': {
            'total_calls': get_counter_with_labels_total(external_api_calls),
            'total_errors': get_counter_with_labels_total(api_errors),
        }
    }
    
    return summary

# Initialize application info
app_info.info({
    'version': '1.0.0',
    'name': 'Airbnb Address Verification System',
    'environment': 'production'
})

# Track application start time for uptime calculation
APP_START_TIME = time.time()

def update_uptime():
    """Update application uptime metric."""
    app_uptime.set(time.time() - APP_START_TIME)