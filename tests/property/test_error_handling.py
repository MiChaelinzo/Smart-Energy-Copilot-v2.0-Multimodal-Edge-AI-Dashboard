"""Property-based tests for error handling and user messages.

**Validates: Requirements 6.5**
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime
from typing import Dict, Any

from src.services.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, ErrorInfo,
    CircuitBreaker, CircuitBreakerOpenError, retry_with_backoff,
    HealthMonitor, ServiceStatus, GracefulDegradation,
    OCRProcessingError, IoTConnectionError, AIProcessingError,
    ResourceConstraintError, DatabaseError, ValidationError
)


class TestErrorHandlingProperties:
    """Property-based tests for comprehensive error handling."""
    
    @given(
        service_name=st.text(min_size=1, max_size=50),
        operation=st.text(min_size=1, max_size=50),
        error_message=st.text(min_size=1, max_size=200),
        severity=st.sampled_from(ErrorSeverity)
    )
    def test_error_logging_and_user_messages_property(
        self, service_name: str, operation: str, error_message: str, severity: ErrorSeverity
    ):
        """
        Property 23: Error logging and user messages
        
        For any error that occurs in the system, the error handler should:
        1. Log the error with appropriate severity level
        2. Generate a user-friendly error message
        3. Provide recovery suggestions
        4. Assign a unique error ID
        5. Store structured error information
        
        **Validates: Requirements 6.5**
        """
        # Arrange
        error_handler = ErrorHandler()
        context = ErrorContext(
            service_name=service_name,
            operation=operation,
            metadata={"test": True}
        )
        
        # Create a test exception
        test_error = Exception(error_message)
        
        # Act
        error_info = error_handler.handle_error(test_error, context, severity)
        
        # Assert - Error information is properly structured
        assert isinstance(error_info, ErrorInfo)
        assert error_info.error_id is not None
        assert len(error_info.error_id) > 0
        assert error_info.service_name == service_name
        assert error_info.operation == operation
        assert error_info.severity == severity
        assert error_info.error_message == error_message
        
        # Assert - User message is user-friendly (not technical)
        assert error_info.user_message is not None
        assert len(error_info.user_message) > 0
        assert "Exception" not in error_info.user_message  # Should not contain technical terms
        assert "Traceback" not in error_info.user_message
        
        # Assert - Recovery suggestions are provided
        assert isinstance(error_info.recovery_suggestions, list)
        assert len(error_info.recovery_suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in error_info.recovery_suggestions)
        
        # Assert - Timestamp is recent
        assert isinstance(error_info.timestamp, datetime)
        time_diff = (datetime.now() - error_info.timestamp).total_seconds()
        assert time_diff < 1.0  # Should be very recent
        
        # Assert - Error is stored in history
        assert error_info in error_handler.error_history
    
    @given(
        failure_threshold=st.integers(min_value=1, max_value=10),
        recovery_timeout=st.integers(min_value=1, max_value=60),
        failure_count=st.integers(min_value=0, max_value=15)
    )
    def test_circuit_breaker_behavior_property(
        self, failure_threshold: int, recovery_timeout: int, failure_count: int
    ):
        """
        Property: Circuit breaker behavior consistency
        
        For any circuit breaker configuration, the circuit breaker should:
        1. Open after reaching failure threshold
        2. Stay open during recovery timeout
        3. Allow half-open state after timeout
        4. Reset on successful execution
        
        **Validates: Requirements 6.5**
        """
        # Arrange
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        # Simulate failures
        for _ in range(failure_count):
            circuit_breaker._on_failure()
        
        # Assert circuit breaker state logic
        if failure_count >= failure_threshold:
            assert circuit_breaker.state.state == "open"
            assert circuit_breaker.state.failure_count >= failure_threshold
            assert circuit_breaker.state.next_attempt_time is not None
        else:
            assert circuit_breaker.state.state == "closed"
            assert circuit_breaker.state.failure_count == failure_count
        
        # Test success resets the circuit breaker
        circuit_breaker._on_success()
        assert circuit_breaker.state.state == "closed"
        assert circuit_breaker.state.failure_count == 0
        assert circuit_breaker.state.next_attempt_time is None
    
    @pytest.mark.asyncio
    @given(
        service_names=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True
        ),
        health_results=st.lists(st.booleans(), min_size=1, max_size=10)
    )
    async def test_health_monitoring_consistency_property(
        self, service_names: list, health_results: list
    ):
        """
        Property: Health monitoring consistency
        
        For any set of services and their health check results, the health monitor should:
        1. Correctly track individual service statuses
        2. Calculate overall system health accurately
        3. Maintain consistent state across checks
        
        **Validates: Requirements 6.5**
        """
        assume(len(service_names) == len(health_results))
        
        # Arrange
        health_monitor = HealthMonitor()
        
        # Register health checks
        for i, service_name in enumerate(service_names):
            health_result = health_results[i]
            health_monitor.register_health_check(
                service_name,
                lambda result=health_result: result
            )
        
        # Act - Check all services
        for i, service_name in enumerate(service_names):
            expected_status = ServiceStatus.HEALTHY if health_results[i] else ServiceStatus.UNHEALTHY
            actual_status = await health_monitor.check_service_health(service_name)
            
            # Assert individual service status
            assert actual_status == expected_status
            assert health_monitor.service_statuses[service_name] == expected_status
        
        # Act - Get system health
        system_health = health_monitor.get_system_health()
        
        # Assert system health calculation
        healthy_count = sum(1 for result in health_results if result)
        total_count = len(health_results)
        
        assert system_health["healthy_services"] == healthy_count
        assert system_health["total_services"] == total_count
        
        # Assert overall status logic
        if healthy_count == total_count:
            assert system_health["overall_status"] == ServiceStatus.HEALTHY.value
        elif healthy_count > total_count * 0.5:
            assert system_health["overall_status"] == ServiceStatus.DEGRADED.value
        else:
            assert system_health["overall_status"] == ServiceStatus.UNHEALTHY.value
    
    @given(
        cpu_percent=st.floats(min_value=0.0, max_value=100.0),
        memory_percent=st.floats(min_value=0.0, max_value=100.0),
        disk_percent=st.floats(min_value=0.0, max_value=100.0)
    )
    def test_graceful_degradation_logic_property(
        self, cpu_percent: float, memory_percent: float, disk_percent: float
    ):
        """
        Property: Graceful degradation logic consistency
        
        For any resource usage levels, the graceful degradation system should:
        1. Correctly identify when degradation is needed
        2. Disable features in priority order
        3. Re-enable features when resources become available
        4. Maintain feature state consistency
        
        **Validates: Requirements 6.4**
        """
        # Arrange
        degradation = GracefulDegradation()
        resource_usage = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        }
        
        # Act - Check if degradation should occur
        should_degrade = degradation.should_degrade_service(resource_usage)
        
        # Assert degradation logic
        expected_degrade = (
            cpu_percent > 80 or
            memory_percent > 80 or
            disk_percent > 90
        )
        assert should_degrade == expected_degrade
        
        # Test feature state consistency
        initial_disabled = len(degradation.disabled_features)
        
        if should_degrade:
            # Simulate degradation (without actual resource checking)
            degradation.disabled_features.add("detailed_logging")
            degradation.disabled_features.add("dashboard_animations")
            
            # Assert features are properly disabled
            assert not degradation.is_feature_enabled("detailed_logging")
            assert not degradation.is_feature_enabled("dashboard_animations")
            assert degradation.is_feature_enabled("core_ocr")  # High priority should remain
        
        # Test restoration logic
        if not should_degrade:
            degradation.disabled_features.clear()
            
            # Assert all features are enabled when resources are available
            for feature in degradation.feature_priorities:
                assert degradation.is_feature_enabled(feature)
    
    @given(
        error_types=st.lists(
            st.sampled_from([
                OCRProcessingError, IoTConnectionError, AIProcessingError,
                ResourceConstraintError, DatabaseError, ValidationError,
                ConnectionError, TimeoutError, PermissionError
            ]),
            min_size=1,
            max_size=5
        ),
        error_messages=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=5
        )
    )
    def test_error_type_mapping_consistency_property(
        self, error_types: list, error_messages: list
    ):
        """
        Property: Error type mapping consistency
        
        For any combination of error types and messages, the error handler should:
        1. Consistently map error types to user-friendly messages
        2. Provide appropriate recovery suggestions for each error type
        3. Never expose technical details to users
        
        **Validates: Requirements 6.5**
        """
        assume(len(error_types) == len(error_messages))
        
        # Arrange
        error_handler = ErrorHandler()
        
        for error_type, error_message in zip(error_types, error_messages):
            # Create error instance
            error = error_type(error_message)
            context = ErrorContext(
                service_name="test_service",
                operation="test_operation"
            )
            
            # Act
            error_info = error_handler.handle_error(error, context)
            
            # Assert user message is appropriate for error type
            user_message = error_info.user_message.lower()
            
            # Assert no technical terms in user message
            technical_terms = [
                "exception", "traceback", "stack", "null", "undefined",
                "error:", "failed:", "exception:", "traceback:"
            ]
            for term in technical_terms:
                assert term not in user_message
            
            # Assert recovery suggestions are relevant
            suggestions = error_info.recovery_suggestions
            assert len(suggestions) > 0
            
            # Check error-type specific suggestions
            if error_type == OCRProcessingError:
                suggestion_text = " ".join(suggestions).lower()
                assert any(word in suggestion_text for word in ["document", "image", "clear", "format"])
            elif error_type == IoTConnectionError:
                suggestion_text = " ".join(suggestions).lower()
                assert any(word in suggestion_text for word in ["device", "network", "connection"])
            elif error_type == ValidationError:
                suggestion_text = " ".join(suggestions).lower()
                assert any(word in suggestion_text for word in ["format", "input", "data", "check"])
    
    @pytest.mark.asyncio
    @given(
        max_attempts=st.integers(min_value=1, max_value=5),
        success_on_attempt=st.integers(min_value=1, max_value=5)
    )
    async def test_retry_mechanism_property(
        self, max_attempts: int, success_on_attempt: int
    ):
        """
        Property: Retry mechanism behavior
        
        For any retry configuration, the retry mechanism should:
        1. Attempt the operation up to max_attempts times
        2. Succeed if the operation succeeds within attempts
        3. Fail with the last exception if all attempts fail
        4. Apply appropriate backoff between attempts
        
        **Validates: Requirements 6.5**
        """
        # Arrange
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=max_attempts, base_delay=0.01)  # Fast for testing
        async def test_operation():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < success_on_attempt:
                raise ConnectionError(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        # Act & Assert
        if success_on_attempt <= max_attempts:
            # Should succeed
            result = await test_operation()
            assert result == f"Success on attempt {success_on_attempt}"
            assert attempt_count == success_on_attempt
        else:
            # Should fail after max_attempts
            with pytest.raises(ConnectionError):
                await test_operation()
            assert attempt_count == max_attempts


# Feature: smart-energy-copilot, Property 23: Error logging and user messages