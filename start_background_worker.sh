#!/bin/bash
# Script to start the Deep Analysis background worker

echo "Starting Deep Analysis background worker..."
python src/background/background_worker.py &
echo "Background worker started with PID: $!"
echo ""
echo "The background worker is now running and will process Deep Analysis jobs."
echo "To check if it's running: ps aux | grep background_worker"
echo "To stop it: kill the PID shown above"