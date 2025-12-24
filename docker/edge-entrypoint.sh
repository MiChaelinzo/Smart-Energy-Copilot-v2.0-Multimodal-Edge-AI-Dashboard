#!/bin/bash
# Edge deployment entrypoint script for Smart Energy Copilot v2.0

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check system resources
check_resources() {
    log "Checking system resources..."
    
    # Check available memory (minimum 1GB required)
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$AVAILABLE_MEM" -lt 1024 ]; then
        log "WARNING: Low memory available: ${AVAILABLE_MEM}MB"
    fi
    
    # Check disk space (minimum 2GB required)
    AVAILABLE_DISK=$(df /app | awk 'NR==2{print $4}')
    if [ "$AVAILABLE_DISK" -lt 2097152 ]; then  # 2GB in KB
        log "WARNING: Low disk space available"
    fi
    
    # Check CPU temperature if available
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
        TEMP_C=$((TEMP / 1000))
        log "CPU Temperature: ${TEMP_C}°C"
        
        if [ "$TEMP_C" -gt 75 ]; then
            log "WARNING: High CPU temperature: ${TEMP_C}°C"
        fi
    fi
}

# Function to initialize database
init_database() {
    log "Initializing database..."
    
    if [ ! -f "$DATA_DIR/energy_copilot.db" ]; then
        log "Creating new database..."
        python -m scripts.init_db
    else
        log "Database already exists, checking for migrations..."
        alembic upgrade head
    fi
}

# Function to optimize for edge deployment
optimize_for_edge() {
    log "Applying edge deployment optimizations..."
    
    # Set memory limits for Python
    export PYTHONMALLOC=malloc
    export MALLOC_TRIM_THRESHOLD_=100000
    
    # Optimize garbage collection for low memory
    export PYTHONGC=1
    
    # Set model quantization flags
    export TORCH_QUANTIZATION=true
    export MODEL_PRECISION=fp16
    
    # Configure for offline operation
    export OFFLINE_MODE=true
    export CLOUD_DEPENDENCIES=false
}

# Function to setup model cache
setup_model_cache() {
    log "Setting up model cache..."
    
    if [ ! -d "$MODEL_CACHE_DIR" ]; then
        mkdir -p "$MODEL_CACHE_DIR"
    fi
    
    # Check if models need to be downloaded/optimized
    if [ ! -f "$MODEL_CACHE_DIR/ernie_quantized.bin" ]; then
        log "Optimizing AI models for edge deployment..."
        python -c "
from src.services.ai_service import ERNIEModelService
service = ERNIEModelService()
service.optimize_for_edge('$MODEL_CACHE_DIR')
"
    fi
}

# Function to start monitoring
start_monitoring() {
    log "Starting system monitoring..."
    
    # Start resource monitoring in background
    python -c "
import asyncio
from src.services.edge_deployment import EdgeDeploymentService
async def monitor():
    service = EdgeDeploymentService(data_dir='$DATA_DIR')
    while True:
        await service.monitor_system_resources()
        await asyncio.sleep(30)
asyncio.run(monitor())
" &
    
    MONITOR_PID=$!
    echo $MONITOR_PID > /tmp/monitor.pid
    log "System monitoring started (PID: $MONITOR_PID)"
}

# Function to handle shutdown
cleanup() {
    log "Shutting down gracefully..."
    
    # Stop monitoring
    if [ -f /tmp/monitor.pid ]; then
        MONITOR_PID=$(cat /tmp/monitor.pid)
        if kill -0 $MONITOR_PID 2>/dev/null; then
            kill $MONITOR_PID
            log "Stopped system monitoring"
        fi
        rm -f /tmp/monitor.pid
    fi
    
    # Flush any pending data
    python -c "
import asyncio
from src.services.edge_deployment import EdgeDeploymentService
async def flush():
    service = EdgeDeploymentService(data_dir='$DATA_DIR')
    if service.is_offline:
        await service._flush_offline_buffer()
asyncio.run(flush())
"
    
    log "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting Smart Energy Copilot v2.0 - Edge Deployment"
    
    # Create required directories
    mkdir -p "$DATA_DIR" "$MODEL_CACHE_DIR" /app/logs /app/updates
    
    # Check system resources
    check_resources
    
    # Apply edge optimizations
    optimize_for_edge
    
    # Initialize database
    init_database
    
    # Setup model cache
    setup_model_cache
    
    # Start monitoring
    start_monitoring
    
    log "Edge deployment initialization complete"
    log "Starting application..."
    
    # Execute the main command
    exec "$@"
}

# Run main function
main "$@"