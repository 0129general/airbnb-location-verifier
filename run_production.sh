#!/bin/bash
# Production deployment script for Replit Autoscale

# Set production environment
export FLASK_ENV=production

# Run gunicorn with production settings
# - Use wsgi:application as the entry point
# - Bind to port 80 for Replit deployment
# - Use 4 workers for better performance
# - Set timeout to 120 seconds for long-running requests
exec gunicorn wsgi:application \
    --bind 0.0.0.0:80 \
    --workers 4 \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    --error-logfile -