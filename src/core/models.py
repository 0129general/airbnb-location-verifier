from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json
import logging
import uuid
import time

# Import metrics for tracking database operations
from core.metrics import db_operations, db_query_time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

db = SQLAlchemy()

class VerificationResult(db.Model):
    """Model for storing verification results with session-based access"""
    __tablename__ = 'verification_results'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    airbnb_url = db.Column(db.Text, nullable=False, index=True)
    extracted_data = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    accessed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Background processing status fields
    processing_status = db.Column(db.String(20), default='pending')  # pending, processing, completed, error
    processing_started_at = db.Column(db.DateTime)
    processing_completed_at = db.Column(db.DateTime)
    processing_error = db.Column(db.Text)
    processing_progress = db.Column(db.Text)  # JSON string with progress updates
    
    @property
    def data(self):
        """Get extracted_data as a dictionary"""
        try:
            return json.loads(self.extracted_data)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    @data.setter
    def data(self, value):
        """Set extracted_data from a dictionary"""
        if isinstance(value, dict):
            self.extracted_data = json.dumps(value, default=str)
        else:
            self.extracted_data = value
    
    @classmethod
    def create_session(cls, airbnb_url, location_data):
        """Create a new session with verification results"""
        session_id = str(uuid.uuid4())
        result = cls(
            session_id=session_id,
            airbnb_url=airbnb_url,
            extracted_data=json.dumps(location_data, default=str)
        )
        return result
    
    @classmethod
    def get_by_session(cls, session_id):
        """Get verification result by session ID and update accessed_at"""
        start_time = time.time()
        try:
            result = cls.query.filter_by(session_id=session_id).first()
            if result:
                result.accessed_at = datetime.utcnow()
                db.session.commit()
                db_operations.labels(operation='read', table='verification_results', status='success').inc()
            else:
                db_operations.labels(operation='read', table='verification_results', status='not_found').inc()
            
            # Track latency
            db_query_time.labels(operation='read', table='verification_results').observe(time.time() - start_time)
            return result
        except Exception as e:
            db_operations.labels(operation='read', table='verification_results', status='error').inc()
            db_query_time.labels(operation='read', table='verification_results').observe(time.time() - start_time)
            logger.error(f"Database error in get_by_session: {e}")
            raise
    
    @classmethod
    def get_cached(cls, airbnb_url, cache_hours=24):
        """Get cached result for the same URL within cache_hours (default: 24 hours)"""
        start_time = time.time()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=cache_hours)
            result = cls.query.filter(
                cls.airbnb_url == airbnb_url,
                cls.created_at > cutoff_time,
                cls.processing_status == 'completed'  # Only return completed results
            ).order_by(cls.created_at.desc()).first()
            
            if result:
                logger.info(f"Found cached result for {airbnb_url} from {result.created_at}")
                db_operations.labels(operation='read', table='verification_results', status='cache_hit').inc()
            else:
                db_operations.labels(operation='read', table='verification_results', status='cache_miss').inc()
            
            # Track query time
            db_query_time.labels(operation='read', table='verification_results').observe(time.time() - start_time)
            
            return result
        except Exception as e:
            db_operations.labels(operation='read', table='verification_results', status='error').inc()
            db_query_time.labels(operation='read', table='verification_results').observe(time.time() - start_time)
            logger.error(f"Database error in get_cached: {e}")
            raise
    
    @classmethod
    def cleanup_expired(cls, expiry_hours=24):
        """Delete sessions older than expiry_hours"""
        start_time = time.time()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=expiry_hours)
            expired = cls.query.filter(cls.accessed_at < cutoff_time).all()
            count = len(expired)
            
            for result in expired:
                db.session.delete(result)
            
            db.session.commit()
            
            # Track cleanup metrics
            if count > 0:
                db_operations.labels(operation='delete', table='verification_results', status='success').inc(count)
                logger.info(f"Cleaned up {count} expired verification results")
            
            # Track query time
            db_query_time.labels(operation='delete', table='verification_results').observe(time.time() - start_time)
            
            return count
        except Exception as e:
            db_operations.labels(operation='delete', table='verification_results', status='error').inc()
            db_query_time.labels(operation='delete', table='verification_results').observe(time.time() - start_time)
            logger.error(f"Database error in cleanup_expired: {e}")
            db.session.rollback()
            raise


class JobQueue(db.Model):
    """Model for managing analysis job queue"""
    __tablename__ = 'job_queue'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    airbnb_url = db.Column(db.Text, nullable=False, index=True)
    
    # Job status and priority
    status = db.Column(db.String(20), nullable=False, default='pending', index=True)
    # Status values: pending, processing, completed, failed, cancelled
    priority = db.Column(db.Integer, default=5)  # 1-10, lower is higher priority
    analysis_mode = db.Column(db.String(20), default='deep')  # quick, deep
    
    # Queue type and batch processing
    queue_type = db.Column(db.String(20), default='deep_analysis', index=True)  # deep_analysis, double_check
    batch_id = db.Column(db.String(36), index=True)  # For grouping batch verification jobs
    
    # Timing
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Results
    verification_result_id = db.Column(db.Integer, db.ForeignKey('verification_results.id'))
    error_message = db.Column(db.Text)
    retry_count = db.Column(db.Integer, default=0)
    max_retries = db.Column(db.Integer, default=3)
    
    # Worker info
    worker_id = db.Column(db.String(50))  # Track which worker is processing
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    
    # User info (for tracking who queued it)
    queued_by = db.Column(db.String(100))  # Could be session ID or user ID
    notes = db.Column(db.Text)  # Optional notes from data entry person
    
    # Relationship to verification results
    verification_result = db.relationship('VerificationResult', backref='job_queue_entry')
    
    @classmethod
    def create_job(cls, airbnb_url, priority=5, analysis_mode='deep', queue_type='deep_analysis', 
                   batch_id=None, queued_by=None, notes=None):
        """Create a new job in the queue"""
        job = cls(
            job_id=str(uuid.uuid4()),
            airbnb_url=airbnb_url,
            priority=priority,
            analysis_mode=analysis_mode,
            queue_type=queue_type,
            batch_id=batch_id,
            queued_by=queued_by,
            notes=notes
        )
        return job
    
    @classmethod
    def get_next_job(cls, worker_id=None):
        """Get the next job to process (highest priority, oldest first)"""
        # Look for pending jobs, ordered by priority and creation time
        job = cls.query.filter_by(status='pending').order_by(
            cls.priority.asc(),  # Lower priority number = higher priority
            cls.created_at.asc()  # Oldest first
        ).first()
        
        if job and worker_id:
            job.status = 'processing'
            job.started_at = datetime.utcnow()
            job.worker_id = worker_id
            job.last_activity = datetime.utcnow()
            db.session.commit()
        
        return job
    
    @classmethod
    def get_queue_stats(cls):
        """Get queue statistics"""
        pending = cls.query.filter_by(status='pending').count()
        processing = cls.query.filter_by(status='processing').count()
        completed_today = cls.query.filter(
            cls.status == 'completed',
            cls.completed_at > datetime.utcnow() - timedelta(hours=24)
        ).count()
        failed = cls.query.filter_by(status='failed').filter(
            cls.retry_count >= cls.max_retries
        ).count()
        
        return {
            'pending': pending,
            'processing': processing,
            'completed_today': completed_today,
            'failed': failed
        }
    
    @classmethod
    def cleanup_stale_jobs(cls, timeout_minutes=30):
        """Mark jobs as failed if they've been processing too long"""
        timeout_cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        stale_jobs = cls.query.filter(
            cls.status == 'processing',
            cls.last_activity < timeout_cutoff
        ).all()
        
        for job in stale_jobs:
            job.status = 'failed'
            job.error_message = f"Job timed out after {timeout_minutes} minutes"
            job.retry_count += 1
            if job.retry_count < job.max_retries:
                job.status = 'pending'  # Retry the job
                job.worker_id = None
                job.started_at = None
        
        db.session.commit()
        return len(stale_jobs)
    
    def mark_completed(self, verification_result_id):
        """Mark job as completed with result"""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.verification_result_id = verification_result_id
        self.last_activity = datetime.utcnow()
        db.session.commit()
    
    def mark_failed(self, error_message):
        """Mark job as failed with error"""
        self.error_message = error_message
        self.retry_count += 1
        self.last_activity = datetime.utcnow()
        
        if self.retry_count < self.max_retries:
            self.status = 'pending'  # Retry
            self.worker_id = None
            self.started_at = None
        else:
            self.status = 'failed'
            self.completed_at = datetime.utcnow()
        
        db.session.commit()