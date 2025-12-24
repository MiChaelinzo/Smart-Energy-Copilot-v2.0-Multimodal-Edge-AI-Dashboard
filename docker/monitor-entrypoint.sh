#!/bin/sh
# Monitor service entrypoint script

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MONITOR: $1"
}

# Create log directory
mkdir -p /app/logs

# Set up log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    # Create logrotate config
    cat > /tmp/logrotate.conf << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 monitor monitor
}

/app/logs/*.jsonl {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 monitor monitor
}
EOF
    
    # Run logrotate in background
    (while true; do
        logrotate -f /tmp/logrotate.conf
        sleep 86400  # Run once per day
    done) &
}

# Function to handle shutdown
cleanup() {
    log "Shutting down monitor service..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    log "Monitor service stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting Edge System Monitor"
    
    # Setup log rotation
    setup_log_rotation
    
    log "Monitor service initialization complete"
    
    # Execute the main command
    exec "$@"
}

# Run main function
main "$@"