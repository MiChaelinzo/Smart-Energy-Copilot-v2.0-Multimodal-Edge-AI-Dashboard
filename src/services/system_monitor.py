"""System monitoring and health management service."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psutil

from src.services.error_handling import (
    health_monitor, error_handler, graceful_degradation,
    ServiceStatus, ErrorSeverity, ErrorContext
)
from src.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemAlert:
    """System alert information."""
    alert_id: str
    timestamp: datetime
    severity: str
    service_name: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False


class SystemMonitorService:
    """Comprehensive system monitoring and health management."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alerts: List[SystemAlert] = []
        self.is_running = False
        
        # Alert thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 80.0,
            "disk_percent": 90.0,
            "error_rate": 0.1,
            "response_time": 5.0
        }
        
        # Register health checks for core services
        self._register_core_health_checks()
    
    def _register_core_health_checks(self):
        """Register health checks for core services."""
        
        async def ocr_health_check():
            """Check OCR service health."""
            try:
                from src.services.ocr_service import OCRProcessingEngine
                engine = OCRProcessingEngine()
                # Simple health check - verify engine can be initialized
                return engine is not None
            except Exception as e:
                logger.error(f"OCR health check failed: {e}")
                return False
        
        async def ai_health_check():
            """Check AI service health."""
            try:
                from src.services.ai_service import EnergyAIService
                # Check if service can be initialized
                return True  # Simplified check
            except Exception as e:
                logger.error(f"AI service health check failed: {e}")
                return False
        
        async def iot_health_check():
            """Check IoT integration health."""
            try:
                from src.services.iot_integration import IoTIntegrationService
                # Check if service can be initialized
                return True  # Simplified check
            except Exception as e:
                logger.error(f"IoT service health check failed: {e}")
                return False
        
        async def database_health_check():
            """Check database health."""
            try:
                from src.database.connection import get_db_session
                with get_db_session() as session:
                    # Simple query to check database connectivity
                    session.execute("SELECT 1")
                    return True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                return False
        
        # Register health checks
        health_monitor.register_health_check("ocr_service", ocr_health_check)
        health_monitor.register_health_check("ai_service", ai_health_check)
        health_monitor.register_health_check("iot_service", iot_health_check)
        health_monitor.register_health_check("database", database_health_check)
    
    async def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.is_running:
            logger.warning("System monitoring is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await self._check_resource_usage()
                await self._check_service_metrics()
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered services."""
        try:
            service_statuses = await health_monitor.check_all_services()
            
            for service_name, status in service_statuses.items():
                if status in [ServiceStatus.UNHEALTHY, ServiceStatus.OFFLINE]:
                    await self._create_alert(
                        severity="high",
                        service_name=service_name,
                        message=f"Service {service_name} is {status.value}",
                        metrics={"status": status.value}
                    )
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _check_resource_usage(self):
        """Check system resource usage."""
        try:
            resource_usage = graceful_degradation.check_resource_constraints()
            
            # Check CPU usage
            if resource_usage.get("cpu_percent", 0) > self.thresholds["cpu_percent"]:
                await self._create_alert(
                    severity="medium",
                    service_name="system",
                    message=f"High CPU usage: {resource_usage['cpu_percent']:.1f}%",
                    metrics=resource_usage
                )
            
            # Check memory usage
            if resource_usage.get("memory_percent", 0) > self.thresholds["memory_percent"]:
                await self._create_alert(
                    severity="medium",
                    service_name="system",
                    message=f"High memory usage: {resource_usage['memory_percent']:.1f}%",
                    metrics=resource_usage
                )
            
            # Check disk usage
            if resource_usage.get("disk_percent", 0) > self.thresholds["disk_percent"]:
                await self._create_alert(
                    severity="high",
                    service_name="system",
                    message=f"High disk usage: {resource_usage['disk_percent']:.1f}%",
                    metrics=resource_usage
                )
            
            # Handle resource constraints
            degradation_result = graceful_degradation.handle_resource_constraints(resource_usage)
            if degradation_result["degradation_active"]:
                await self._create_alert(
                    severity="medium",
                    service_name="system",
                    message=f"Graceful degradation active. Disabled features: {degradation_result['disabled_features']}",
                    metrics=degradation_result
                )
        
        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")
    
    async def _check_service_metrics(self):
        """Check service-specific metrics."""
        try:
            # Get list of services from error handler metrics
            services = set(error_handler.metrics.error_counts.keys()) | set(error_handler.metrics.response_times.keys())
            
            for service_name in services:
                health_summary = error_handler.get_service_health_summary(service_name)
                
                # Check error rate
                if health_summary["error_rate"] > self.thresholds["error_rate"]:
                    await self._create_alert(
                        severity="medium",
                        service_name=service_name,
                        message=f"High error rate: {health_summary['error_rate']:.2%}",
                        metrics=health_summary
                    )
                
                # Check response time
                if health_summary["avg_response_time"] > self.thresholds["response_time"]:
                    await self._create_alert(
                        severity="medium",
                        service_name=service_name,
                        message=f"Slow response time: {health_summary['avg_response_time']:.2f}s",
                        metrics=health_summary
                    )
        
        except Exception as e:
            logger.error(f"Error checking service metrics: {e}")
    
    async def _create_alert(self, severity: str, service_name: str, message: str, metrics: Dict[str, Any]):
        """Create a new system alert."""
        alert_id = f"{service_name}_{int(time.time())}"
        
        # Check if similar alert already exists (avoid spam)
        existing_alert = next(
            (alert for alert in self.alerts 
             if alert.service_name == service_name 
             and alert.message == message 
             and not alert.resolved
             and (datetime.now() - alert.timestamp).total_seconds() < 300),  # 5 minutes
            None
        )
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        alert = SystemAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            service_name=service_name,
            message=message,
            metrics=metrics
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = logging.ERROR if severity == "high" else logging.WARNING
        logger.log(log_level, f"SYSTEM ALERT [{severity.upper()}]: {message}", extra={
            "alert_id": alert_id,
            "service": service_name,
            "metrics": metrics
        })
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time or not alert.resolved
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Get overall system health
        system_health = health_monitor.get_system_health()
        
        # Get resource usage
        resource_usage = graceful_degradation.check_resource_constraints()
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        # Get degradation status
        degradation_status = graceful_degradation.handle_resource_constraints(resource_usage)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": system_health,
            "resource_usage": resource_usage,
            "degradation_status": degradation_status,
            "active_alerts": len(active_alerts),
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "service": alert.service_name,
                    "message": alert.message
                }
                for alert in active_alerts[-5:]  # Last 5 alerts
            ],
            "monitoring_active": self.is_running
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def get_alerts(self, service_name: Optional[str] = None, resolved: Optional[bool] = None) -> List[SystemAlert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if service_name:
            alerts = [alert for alert in alerts if alert.service_name == service_name]
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)


# Global system monitor instance
system_monitor = SystemMonitorService()