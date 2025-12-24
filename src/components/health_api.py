"""System health monitoring API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.services.system_monitor import system_monitor
from src.services.error_handling import error_handler, health_monitor, graceful_degradation
from src.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    try:
        return system_monitor.get_system_status()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@router.get("/services")
async def get_service_health() -> Dict[str, Any]:
    """Get health status of all registered services."""
    try:
        service_statuses = await health_monitor.check_all_services()
        overall_health = health_monitor.get_system_health()
        
        return {
            "overall_health": overall_health,
            "services": {
                name: {
                    "status": status.value,
                    "health_summary": error_handler.get_service_health_summary(name)
                }
                for name, status in service_statuses.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting service health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service health")


@router.get("/alerts")
async def get_alerts(
    service_name: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get system alerts with optional filtering."""
    try:
        alerts = system_monitor.get_alerts(service_name=service_name, resolved=resolved)
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "service_name": alert.service_name,
                    "message": alert.message,
                    "resolved": alert.resolved,
                    "metrics": alert.metrics
                }
                for alert in alerts
            ],
            "total_count": len(alerts),
            "filters": {
                "service_name": service_name,
                "resolved": resolved,
                "limit": limit
            }
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolve a specific alert."""
    try:
        success = system_monitor.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or already resolved")
        
        return {
            "success": True,
            "message": f"Alert {alert_id} resolved successfully",
            "resolved_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/resources")
async def get_resource_usage() -> Dict[str, Any]:
    """Get current system resource usage and degradation status."""
    try:
        resource_usage = graceful_degradation.check_resource_constraints()
        degradation_status = graceful_degradation.handle_resource_constraints(resource_usage)
        
        return {
            "resource_usage": resource_usage,
            "degradation_status": degradation_status,
            "thresholds": {
                "cpu_warning": 80.0,
                "memory_warning": 80.0,
                "disk_warning": 90.0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resource usage")


@router.get("/errors")
async def get_error_history(
    service_name: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get recent error history with optional filtering."""
    try:
        errors = error_handler.error_history
        
        # Apply filters
        if service_name:
            errors = [error for error in errors if error.service_name == service_name]
        
        if severity:
            errors = [error for error in errors if error.severity.value == severity]
        
        # Sort by timestamp (most recent first) and limit
        errors = sorted(errors, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return {
            "errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp.isoformat(),
                    "severity": error.severity.value,
                    "service_name": error.service_name,
                    "operation": error.operation,
                    "error_type": error.error_type,
                    "user_message": error.user_message,
                    "recovery_suggestions": error.recovery_suggestions
                }
                for error in errors
            ],
            "total_count": len(errors),
            "filters": {
                "service_name": service_name,
                "severity": severity,
                "limit": limit
            }
        }
    except Exception as e:
        logger.error(f"Error getting error history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve error history")


@router.get("/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Get detailed system performance metrics."""
    try:
        metrics = error_handler.metrics
        
        return {
            "error_counts": metrics.error_counts,
            "response_times": {
                service: {
                    "average": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0,
                    "count": len(times)
                }
                for service, times in metrics.response_times.items()
            },
            "resource_usage": metrics.resource_usage,
            "last_health_check": metrics.last_health_check.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.post("/monitoring/start")
async def start_monitoring() -> Dict[str, Any]:
    """Start system monitoring."""
    try:
        await system_monitor.start_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring started",
            "started_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@router.post("/monitoring/stop")
async def stop_monitoring() -> Dict[str, Any]:
    """Stop system monitoring."""
    try:
        await system_monitor.stop_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring stopped",
            "stopped_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")


@router.get("/circuit-breakers")
async def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers (if any are configured)."""
    try:
        # This would be expanded if circuit breakers are implemented per service
        return {
            "circuit_breakers": {},
            "message": "Circuit breaker status tracking not yet implemented per service",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve circuit breaker status")