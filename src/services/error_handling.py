"""Comprehensive error handling and resilience utilities."""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    service_name: str
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    service_name: str
    operation: str
    error_type: str
    error_message: str
    user_message: str
    recovery_suggestions: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    next_attempt_time: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.state = CircuitBreakerState()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker pattern."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._call(func, *args, **kwargs)
        return wrapper
    
    async def _call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker logic."""
        if self.state.state == "open":
            if self._should_attempt_reset():
                self.state.state = "half_open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.state.next_attempt_time and 
            datetime.now() >= self.state.next_attempt_time
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.state.failure_count = 0
        self.state.state = "closed"
        self.state.next_attempt_time = None
    
    def _on_failure(self):
        """Handle failed execution."""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "open"
            self.state.next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s",
                        error=str(e),
                        function=func.__name__
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.user_message_templates = {
            "ocr_processing_error": "We encountered an issue processing your document. Please ensure the image is clear and try again.",
            "iot_connection_error": "Unable to connect to your IoT device. Please check the device status and network connection.",
            "ai_processing_error": "Our AI analysis is temporarily unavailable. We'll retry automatically.",
            "database_error": "We're experiencing database issues. Your data is safe and we're working to resolve this.",
            "resource_constraint": "System resources are currently limited. Some features may be temporarily unavailable.",
            "network_error": "Network connectivity issues detected. Operating in offline mode.",
            "validation_error": "The provided data doesn't meet our requirements. Please check and try again.",
            "authentication_error": "Authentication failed. Please check your credentials.",
            "permission_error": "You don't have permission to perform this action.",
            "rate_limit_error": "Too many requests. Please wait a moment before trying again."
        }
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ErrorInfo:
        """Handle and log error with context."""
        error_id = f"{context.service_name}_{int(time.time())}"
        error_type = type(error).__name__
        
        # Determine user-friendly message
        user_message = self._get_user_message(error_type, str(error))
        
        # Generate recovery suggestions
        recovery_suggestions = self._get_recovery_suggestions(error_type, context)
        
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            service_name=context.service_name,
            operation=context.operation,
            error_type=error_type,
            error_message=str(error),
            user_message=user_message,
            recovery_suggestions=recovery_suggestions,
            context=context.metadata,
            stack_trace=self._get_stack_trace(error) if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        )
        
        # Log error
        self._log_error(error_info)
        
        # Store in history
        self.error_history.append(error_info)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        return error_info
    
    def _get_user_message(self, error_type: str, error_message: str) -> str:
        """Generate user-friendly error message."""
        # Map error types to user message templates
        error_mapping = {
            "ConnectionError": "network_error",
            "TimeoutError": "network_error",
            "ValidationError": "validation_error",
            "PermissionError": "permission_error",
            "AuthenticationError": "authentication_error",
            "RateLimitError": "rate_limit_error",
            "DatabaseError": "database_error",
            "ResourceConstraintError": "resource_constraint",
            "OCRProcessingError": "ocr_processing_error",
            "IoTConnectionError": "iot_connection_error",
            "AIProcessingError": "ai_processing_error"
        }
        
        template_key = error_mapping.get(error_type, "general_error")
        return self.user_message_templates.get(
            template_key,
            "An unexpected error occurred. Our team has been notified and is working to resolve this."
        )
    
    def _get_recovery_suggestions(self, error_type: str, context: ErrorContext) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = {
            "ConnectionError": [
                "Check your internet connection",
                "Verify device network settings",
                "Try again in a few moments"
            ],
            "ValidationError": [
                "Check the format of your input data",
                "Ensure all required fields are provided",
                "Refer to the documentation for valid formats"
            ],
            "OCRProcessingError": [
                "Ensure the document image is clear and well-lit",
                "Try uploading a higher resolution image",
                "Check that the document format is supported (PDF, JPEG, PNG)"
            ],
            "IoTConnectionError": [
                "Check that your IoT device is powered on",
                "Verify network connectivity to the device",
                "Check device configuration settings"
            ],
            "ResourceConstraintError": [
                "Wait for system resources to become available",
                "Try processing smaller amounts of data",
                "Contact support if the issue persists"
            ]
        }
        
        return suggestions.get(error_type, ["Try the operation again", "Contact support if the issue persists"])
    
    def _get_stack_trace(self, error: Exception) -> str:
        """Get formatted stack trace."""
        import traceback
        return traceback.format_exc()
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_data = {
            "error_id": error_info.error_id,
            "service": error_info.service_name,
            "operation": error_info.operation,
            "error_type": error_info.error_type,
            "error_message": error_info.error_message,
            "severity": error_info.severity.value,
            "context": error_info.context
        }
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.service_statuses: Dict[str, ServiceStatus] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 5.0,  # 5 seconds
            "memory_usage": 0.8,  # 80% memory usage
            "cpu_usage": 0.8  # 80% CPU usage
        }
        
    def register_health_check(self, service_name: str, check_func: Callable):
        """Register a health check function for a service."""
        self.health_checks[service_name] = check_func
        self.service_statuses[service_name] = ServiceStatus.HEALTHY
    
    async def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check health of a specific service."""
        if service_name not in self.health_checks:
            return ServiceStatus.OFFLINE
        
        try:
            health_check = self.health_checks[service_name]
            is_healthy = await health_check() if asyncio.iscoroutinefunction(health_check) else health_check()
            
            status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            self.service_statuses[service_name] = status
            return status
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            self.service_statuses[service_name] = ServiceStatus.OFFLINE
            return ServiceStatus.OFFLINE
    
    async def check_all_services(self) -> Dict[str, ServiceStatus]:
        """Check health of all registered services."""
        results = {}
        for service_name in self.health_checks:
            results[service_name] = await self.check_service_health(service_name)
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_services = sum(1 for status in self.service_statuses.values() if status == ServiceStatus.HEALTHY)
        total_services = len(self.service_statuses)
        
        if total_services == 0:
            overall_status = ServiceStatus.OFFLINE
        elif healthy_services == total_services:
            overall_status = ServiceStatus.HEALTHY
        elif healthy_services > total_services * 0.5:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.UNHEALTHY
        
        return {
            "overall_status": overall_status.value,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "service_statuses": {name: status.value for name, status in self.service_statuses.items()},
            "timestamp": datetime.now().isoformat()
        }


class GracefulDegradation:
    """Graceful degradation for resource constraints."""
    
    def __init__(self):
        self.feature_priorities = {
            "core_ocr": 1,
            "basic_ai_analysis": 2,
            "iot_data_collection": 3,
            "advanced_recommendations": 4,
            "real_time_updates": 5,
            "dashboard_animations": 6,
            "detailed_logging": 7
        }
        self.disabled_features: set = set()
        
    def check_resource_constraints(self) -> Dict[str, float]:
        """Check current resource usage."""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    def should_degrade_service(self, resource_usage: Dict[str, float]) -> bool:
        """Determine if service should be degraded."""
        return (
            resource_usage.get("cpu_percent", 0) > 80 or
            resource_usage.get("memory_percent", 0) > 80 or
            resource_usage.get("disk_percent", 0) > 90
        )
    
    def handle_resource_constraints(self, resource_usage: Dict[str, float]) -> Dict[str, Any]:
        """Handle resource constraints and return structured result."""
        should_degrade = self.should_degrade_service(resource_usage)
        
        if should_degrade:
            self.degrade_services(resource_usage)
        else:
            self.restore_services()
        
        return {
            "degradation_active": should_degrade,
            "disabled_features": list(self.disabled_features),
            "core_functions": {
                "enabled": True,  # Core functions are always enabled
                "features": ["core_ocr", "basic_ai_analysis", "iot_data_collection"]
            },
            "resource_usage": resource_usage,
            "available_features": [
                feature for feature in self.feature_priorities.keys() 
                if feature not in self.disabled_features
            ]
        }
    
    def degrade_services(self, resource_usage: Dict[str, float]):
        """Degrade services based on resource constraints."""
        if not self.should_degrade_service(resource_usage):
            return
        
        # Disable features in reverse priority order
        sorted_features = sorted(
            self.feature_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for feature, priority in sorted_features:
            if feature not in self.disabled_features:
                self.disabled_features.add(feature)
                logger.warning(f"Disabling feature '{feature}' due to resource constraints")
                
                # Re-check after each feature disabled
                new_usage = self.check_resource_constraints()
                if not self.should_degrade_service(new_usage):
                    break
    
    def restore_services(self):
        """Restore services when resources become available."""
        resource_usage = self.check_resource_constraints()
        
        if self.should_degrade_service(resource_usage):
            return  # Still under resource pressure
        
        # Re-enable features in priority order
        sorted_features = sorted(
            self.feature_priorities.items(),
            key=lambda x: x[1]
        )
        
        for feature, priority in sorted_features:
            if feature in self.disabled_features:
                self.disabled_features.remove(feature)
                logger.info(f"Re-enabling feature '{feature}' - resources available")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is currently enabled."""
        return feature not in self.disabled_features


# Global instances
error_handler = ErrorHandler()
health_monitor = HealthMonitor()
graceful_degradation = GracefulDegradation()


# Custom exceptions
class OCRProcessingError(Exception):
    """OCR processing specific error."""
    pass


class IoTConnectionError(Exception):
    """IoT connection specific error."""
    pass


class AIProcessingError(Exception):
    """AI processing specific error."""
    pass


class ResourceConstraintError(Exception):
    """Resource constraint specific error."""
    pass


class DatabaseError(Exception):
    """Database operation specific error."""
    pass


class ValidationError(Exception):
    """Data validation specific error."""
    pass


class AuthenticationError(Exception):
    """Authentication specific error."""
    pass


class RateLimitError(Exception):
    """Rate limiting specific error."""
    pass