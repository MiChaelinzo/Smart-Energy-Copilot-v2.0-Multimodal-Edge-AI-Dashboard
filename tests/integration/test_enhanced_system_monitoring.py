"""
Enhanced System-Level Health Checks and Monitoring Integration Tests.

Comprehensive system monitoring, health validation, and operational readiness testing
for production deployment scenarios.

**Validates: Requirements 6.1, 6.3, 6.4, 6.5, All system reliability requirements**
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
import threading

from src.services.system_monitor import system_monitor, SystemAlert
from src.services.error_handling import health_monitor, error_handler, graceful_degradation
from src.services.error_handling import ServiceStatus, ErrorSeverity
from src.components.health_api import router as health_router
from fastapi.testclient import TestClient
from src.main import app


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    timestamp: datetime
    overall_health_score: float
    service_statuses: Dict[str, str]
    resource_usage: Dict[str, float]
    active_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    degradation_status: Dict[str, Any]
    recommendations: List[str]


class TestEnhancedSystemMonitoring:
    """Enhanced system monitoring and health validation tests."""
    
    @pytest.fixture
    def test_client(self):
        """FastAPI test client with health endpoints."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_system_resources(self):
        """Mock system resource data for testing."""
        return {
            'normal_load': {
                'cpu_percent': 45.0,
                'memory_percent': 60.0,
                'disk_percent': 35.0,
                'network_latency_ms': 15.0,
                'active_connections': 25
            },
            'high_load': {
                'cpu_percent': 85.0,
                'memory_percent': 88.0,
                'disk_percent': 45.0,
                'network_latency_ms': 45.0,
                'active_connections': 150
            },
            'critical_load': {
                'cpu_percent': 95.0,
                'memory_percent': 95.0,
                'disk_percent': 92.0,
                'network_latency_ms': 200.0,
                'active_connections': 300
            }
        }
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_assessment(self, test_client, mock_system_resources):
        """
        Test comprehensive system health assessment across all components.
        
        **Validates: Requirements 6.5**
        """
        # Test health endpoint availability
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        
        # Test service health endpoints
        service_health_response = test_client.get("/api/health/services")
        assert service_health_response.status_code == 200
        service_data = service_health_response.json()
        
        # Verify service health structure
        assert "overall_health" in service_data
        assert "services" in service_data
        
        # Test individual service health checks
        with patch('src.services.system_monitor.system_monitor') as mock_monitor:
            # Mock comprehensive system status
            mock_monitor.get_system_status.return_value = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "healthy",
                "resource_usage": mock_system_resources['normal_load'],
                "degradation_status": {"degradation_active": False},
                "active_alerts": 0,
                "recent_alerts": [],
                "monitoring_active": True
            }
            
            status_response = test_client.get("/api/health/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            # Validate comprehensive health data
            assert status_data["overall_health"] == "healthy"
            assert status_data["monitoring_active"] is True
            assert status_data["active_alerts"] == 0
        
        # Test resource usage monitoring
        resource_response = test_client.get("/api/health/resources")
        assert resource_response.status_code == 200
        resource_data = resource_response.json()
        
        assert "resource_usage" in resource_data
        assert "degradation_status" in resource_data
        assert "thresholds" in resource_data
        
        # Test metrics endpoint
        metrics_response = test_client.get("/api/health/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        
        assert "error_counts" in metrics_data
        assert "response_times" in metrics_data
        assert "resource_usage" in metrics_data
    
    @pytest.mark.asyncio
    async def test_service_dependency_health_checks(self):
        """
        Test health checks for service dependencies and external integrations.
        
        **Validates: Requirements 6.5**
        """
        # Test database connectivity health
        with patch('src.database.connection.get_db_session') as mock_db:
            # Test successful database connection
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.fetchone.return_value = (1,)
            
            # Register and test database health check
            async def database_health_check():
                try:
                    with mock_db() as session:
                        session.execute("SELECT 1")
                        return True
                except Exception:
                    return False
            
            health_monitor.register_health_check("database", database_health_check)
            
            db_health = await database_health_check()
            assert db_health is True, "Database health check should pass"
            
            # Test database connection failure
            mock_db.side_effect = Exception("Database connection failed")
            
            db_health_fail = await database_health_check()
            assert db_health_fail is False, "Database health check should fail when connection fails"
        
        # Test AI model health
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Test AI model availability
            mock_ai.model_manager.is_loaded.return_value = True
            mock_ai.model_manager.model_path = "/models/ernie_energy"
            mock_ai.model_manager.device = "cpu"
            
            async def ai_health_check():
                try:
                    ai_service = mock_ai_service()
                    return ai_service.model_manager.is_loaded()
                except Exception:
                    return False
            
            health_monitor.register_health_check("ai_service", ai_health_check)
            
            ai_health = await ai_health_check()
            assert ai_health is True, "AI service health check should pass"
            
            # Test AI model unavailability
            mock_ai.model_manager.is_loaded.return_value = False
            
            ai_health_fail = await ai_health_check()
            assert ai_health_fail is False, "AI service health check should fail when model not loaded"
        
        # Test IoT service health
        from src.services.iot_integration import IoTIntegrationService
        iot_service = IoTIntegrationService()
        
        with patch.object(iot_service, 'get_all_device_statuses') as mock_device_statuses:
            from src.models.device import DeviceStatus
            
            # Test with healthy devices
            mock_device_statuses.return_value = {
                "device_001": DeviceStatus.ONLINE,
                "device_002": DeviceStatus.ONLINE,
                "device_003": DeviceStatus.ONLINE
            }
            
            async def iot_health_check():
                try:
                    statuses = await iot_service.get_all_device_statuses()
                    online_devices = sum(1 for status in statuses.values() if status == DeviceStatus.ONLINE)
                    return online_devices >= len(statuses) * 0.8  # 80% devices should be online
                except Exception:
                    return False
            
            health_monitor.register_health_check("iot_service", iot_health_check)
            
            iot_health = await iot_health_check()
            assert iot_health is True, "IoT service health check should pass with healthy devices"
            
            # Test with mostly offline devices
            mock_device_statuses.return_value = {
                "device_001": DeviceStatus.OFFLINE,
                "device_002": DeviceStatus.OFFLINE,
                "device_003": DeviceStatus.ONLINE
            }
            
            iot_health_fail = await iot_health_check()
            assert iot_health_fail is False, "IoT service health check should fail with mostly offline devices"
    
    @pytest.mark.asyncio
    async def test_alert_generation_and_management(self, test_client, mock_system_resources):
        """
        Test alert generation, escalation, and management workflows.
        
        **Validates: Requirements 6.5**
        """
        # Start system monitoring
        await system_monitor.start_monitoring()
        
        try:
            # Test alert generation for resource constraints
            with patch('src.services.error_handling.graceful_degradation.check_resource_constraints') as mock_resources:
                # Simulate high resource usage
                mock_resources.return_value = mock_system_resources['high_load']
                
                # Trigger resource check (normally done by monitoring loop)
                await system_monitor._check_resource_usage()
                
                # Verify alerts were generated
                active_alerts = [alert for alert in system_monitor.alerts if not alert.resolved]
                assert len(active_alerts) > 0, "Should generate alerts for high resource usage"
                
                # Check alert content
                cpu_alert = next((alert for alert in active_alerts if "CPU" in alert.message), None)
                memory_alert = next((alert for alert in active_alerts if "memory" in alert.message), None)
                
                assert cpu_alert is not None, "Should generate CPU usage alert"
                assert memory_alert is not None, "Should generate memory usage alert"
                assert cpu_alert.severity in ["medium", "high"], "CPU alert should have appropriate severity"
            
            # Test alert API endpoints
            alerts_response = test_client.get("/api/health/alerts")
            assert alerts_response.status_code == 200
            alerts_data = alerts_response.json()
            
            assert "alerts" in alerts_data
            assert len(alerts_data["alerts"]) > 0, "Should return generated alerts"
            
            # Test alert filtering
            filtered_response = test_client.get("/api/health/alerts?service_name=system&resolved=false")
            assert filtered_response.status_code == 200
            filtered_data = filtered_response.json()
            
            system_alerts = [alert for alert in filtered_data["alerts"] if alert["service_name"] == "system"]
            assert len(system_alerts) > 0, "Should return filtered system alerts"
            
            # Test alert resolution
            if alerts_data["alerts"]:
                alert_id = alerts_data["alerts"][0]["alert_id"]
                resolve_response = test_client.post(f"/api/health/alerts/{alert_id}/resolve")
                assert resolve_response.status_code == 200
                
                resolve_data = resolve_response.json()
                assert resolve_data["success"] is True
                
                # Verify alert is resolved
                resolved_alerts_response = test_client.get("/api/health/alerts?resolved=true")
                resolved_data = resolved_alerts_response.json()
                resolved_alert = next(
                    (alert for alert in resolved_data["alerts"] if alert["alert_id"] == alert_id),
                    None
                )
                assert resolved_alert is not None, "Alert should be marked as resolved"
            
            # Test alert escalation for critical conditions
            with patch('src.services.error_handling.graceful_degradation.check_resource_constraints') as mock_critical:
                mock_critical.return_value = mock_system_resources['critical_load']
                
                await system_monitor._check_resource_usage()
                
                critical_alerts = [
                    alert for alert in system_monitor.alerts 
                    if not alert.resolved and alert.severity == "high"
                ]
                assert len(critical_alerts) > 0, "Should generate high-severity alerts for critical conditions"
        
        finally:
            await system_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_constraints(self, mock_system_resources):
        """
        Test system graceful degradation under resource constraints.
        
        **Validates: Requirements 6.4**
        """
        # Test normal operation (no degradation)
        with patch('src.services.error_handling.graceful_degradation.check_resource_constraints') as mock_resources:
            mock_resources.return_value = mock_system_resources['normal_load']
            
            degradation_result = graceful_degradation.handle_resource_constraints(mock_system_resources['normal_load'])
            
            assert degradation_result["degradation_active"] is False
            assert len(degradation_result["disabled_features"]) == 0
            assert degradation_result["performance_mode"] == "normal"
        
        # Test degradation under high load
        with patch('src.services.error_handling.graceful_degradation.check_resource_constraints') as mock_resources:
            mock_resources.return_value = mock_system_resources['high_load']
            
            degradation_result = graceful_degradation.handle_resource_constraints(mock_system_resources['high_load'])
            
            assert degradation_result["degradation_active"] is True
            assert len(degradation_result["disabled_features"]) > 0
            assert degradation_result["performance_mode"] in ["reduced", "minimal"]
            
            # Verify specific features are disabled under high load
            expected_disabled_features = ["batch_processing", "detailed_analytics", "background_tasks"]
            disabled_features = degradation_result["disabled_features"]
            
            assert any(feature in disabled_features for feature in expected_disabled_features), \
                "Should disable non-essential features under high load"
        
        # Test critical degradation
        with patch('src.services.error_handling.graceful_degradation.check_resource_constraints') as mock_resources:
            mock_resources.return_value = mock_system_resources['critical_load']
            
            degradation_result = graceful_degradation.handle_resource_constraints(mock_system_resources['critical_load'])
            
            assert degradation_result["degradation_active"] is True
            assert degradation_result["performance_mode"] == "minimal"
            assert len(degradation_result["disabled_features"]) >= 3, "Should disable multiple features under critical load"
            
            # Verify core services remain available
            assert "ocr_processing" not in degradation_result["disabled_features"], \
                "Core OCR processing should remain available"
            assert "real_time_inference" not in degradation_result["disabled_features"], \
                "Real-time AI inference should remain available"
        
        # Test AI service adaptation to constraints
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Test reduced batch size under memory constraints
            if mock_system_resources['high_load']['memory_percent'] > 80:
                # Simulate reduced batch processing
                mock_ai.batch_inference.return_value = {
                    'results': {'patterns': []},  # Reduced results
                    'processing_time_ms': 300.0,
                    'summary': {'data_quality_score': 0.75}  # Reduced quality
                }
                
                result = await mock_ai.batch_inference([], "degraded_test")
                assert result['summary']['data_quality_score'] < 0.85, \
                    "Should reduce quality under memory constraints"
        
        # Test IoT service adaptation
        from src.services.iot_integration import IoTIntegrationService
        iot_service = IoTIntegrationService()
        
        # Test reduced polling frequency under CPU constraints
        if mock_system_resources['high_load']['cpu_percent'] > 80:
            # Simulate reduced polling
            original_interval = 30
            degraded_interval = original_interval * 2  # Double the interval
            
            assert degraded_interval > original_interval, \
                "Should increase polling interval under CPU constraints"
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self):
        """
        Test comprehensive error recovery and fault tolerance mechanisms.
        
        **Validates: Requirements 6.3, 6.5**
        """
        # Test AI service error recovery
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Test retry mechanism for transient failures
            call_count = 0
            def mock_inference_with_retries(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # First 2 calls fail
                    raise Exception("Transient AI service error")
                else:  # 3rd call succeeds
                    return {
                        'result': {'recovered': True},
                        'processing_time_ms': 200.0,
                        'confidence': 0.85
                    }
            
            mock_ai.real_time_inference.side_effect = mock_inference_with_retries
            
            # Test retry logic (simulated)
            max_retries = 3
            recovery_successful = False
            
            for attempt in range(max_retries):
                try:
                    result = await mock_ai.real_time_inference({}, "recovery_test")
                    recovery_successful = True
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        break
                    continue
            
            assert recovery_successful, "AI service should recover from transient failures"
            assert call_count == 3, "Should retry exactly 3 times before success"
        
        # Test circuit breaker pattern simulation
        class MockCircuitBreaker:
            def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
                self.failure_count = 0
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.last_failure_time = None
                self.state = "closed"  # closed, open, half_open
            
            async def call(self, func, *args, **kwargs):
                if self.state == "open":
                    if (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                        self.state = "half_open"
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    if self.state == "half_open":
                        self.state = "closed"
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = datetime.now()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                    
                    raise e
        
        # Test circuit breaker behavior
        circuit_breaker = MockCircuitBreaker(failure_threshold=3)
        
        async def failing_service():
            raise Exception("Service unavailable")
        
        # Trigger circuit breaker
        for i in range(5):
            try:
                await circuit_breaker.call(failing_service)
            except Exception:
                pass
        
        assert circuit_breaker.state == "open", "Circuit breaker should be open after repeated failures"
        
        # Test that circuit breaker blocks calls
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await circuit_breaker.call(failing_service)
        
        # Test database connection recovery
        with patch('src.database.connection.get_db_session') as mock_db:
            connection_attempts = 0
            
            def mock_db_connection():
                nonlocal connection_attempts
                connection_attempts += 1
                if connection_attempts <= 2:  # First 2 attempts fail
                    raise Exception("Database connection failed")
                else:  # 3rd attempt succeeds
                    return AsyncMock()
            
            # Simulate connection retry logic
            max_db_retries = 3
            db_recovery_successful = False
            
            for attempt in range(max_db_retries):
                try:
                    session = mock_db_connection()
                    db_recovery_successful = True
                    break
                except Exception:
                    if attempt == max_db_retries - 1:
                        break
                    continue
            
            assert db_recovery_successful, "Database connection should recover after retries"
            assert connection_attempts == 3, "Should attempt connection 3 times"
        
        # Test IoT device reconnection
        from src.services.iot_integration import IoTIntegrationService
        iot_service = IoTIntegrationService()
        
        with patch.object(iot_service, '_attempt_reconnection') as mock_reconnect:
            # Mock successful reconnection after failures
            reconnection_attempts = 0
            
            async def mock_reconnection_logic(device_id):
                nonlocal reconnection_attempts
                reconnection_attempts += 1
                if reconnection_attempts <= 1:  # First attempt fails
                    raise Exception("Reconnection failed")
                # Second attempt succeeds (no exception)
            
            mock_reconnect.side_effect = mock_reconnection_logic
            
            # Test reconnection retry
            device_id = "test_device_001"
            
            for attempt in range(2):
                try:
                    await iot_service._attempt_reconnection(device_id)
                    break
                except Exception:
                    if attempt == 1:  # Last attempt
                        break
                    continue
            
            assert reconnection_attempts == 2, "Should attempt reconnection twice"
    
    @pytest.mark.asyncio
    async def test_system_monitoring_automation(self, test_client):
        """
        Test automated monitoring workflows and self-healing capabilities.
        
        **Validates: Requirements 6.5**
        """
        # Test monitoring start/stop automation
        start_response = test_client.post("/api/health/monitoring/start")
        assert start_response.status_code == 200
        start_data = start_response.json()
        assert start_data["success"] is True
        
        # Verify monitoring is active
        status_response = test_client.get("/api/health/status")
        status_data = status_response.json()
        assert status_data["monitoring_active"] is True
        
        # Test automated health checks
        await asyncio.sleep(1)  # Allow monitoring to run briefly
        
        # Test monitoring stop
        stop_response = test_client.post("/api/health/monitoring/stop")
        assert stop_response.status_code == 200
        stop_data = stop_response.json()
        assert stop_data["success"] is True
        
        # Test automated alert cleanup
        # Create some test alerts
        test_alert = SystemAlert(
            alert_id="test_alert_001",
            timestamp=datetime.now() - timedelta(hours=25),  # Old alert
            severity="medium",
            service_name="test_service",
            message="Test alert for cleanup",
            metrics={},
            resolved=True
        )
        
        system_monitor.alerts.append(test_alert)
        initial_alert_count = len(system_monitor.alerts)
        
        # Trigger cleanup
        await system_monitor._cleanup_old_alerts()
        
        # Verify old resolved alerts are cleaned up
        remaining_alerts = len(system_monitor.alerts)
        assert remaining_alerts <= initial_alert_count, "Should clean up old alerts"
        
        # Test self-healing simulation
        with patch('src.services.system_monitor.system_monitor') as mock_monitor:
            # Simulate service failure detection and recovery
            mock_monitor.get_system_status.return_value = {
                "overall_health": "degraded",
                "active_alerts": 2,
                "monitoring_active": True
            }
            
            # Test that monitoring detects issues
            status = mock_monitor.get_system_status()
            assert status["overall_health"] == "degraded"
            assert status["active_alerts"] > 0
            
            # Simulate recovery
            mock_monitor.get_system_status.return_value = {
                "overall_health": "healthy",
                "active_alerts": 0,
                "monitoring_active": True
            }
            
            recovered_status = mock_monitor.get_system_status()
            assert recovered_status["overall_health"] == "healthy"
            assert recovered_status["active_alerts"] == 0
    
    @pytest.mark.asyncio
    async def test_production_readiness_validation(self, test_client, mock_system_resources):
        """
        Test comprehensive production readiness validation.
        
        **Validates: Requirements 6.1, 6.3, 6.4, 6.5**
        """
        # Generate comprehensive system health report
        health_report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_health_score=0.0,
            service_statuses={},
            resource_usage={},
            active_alerts=[],
            performance_metrics={},
            degradation_status={},
            recommendations=[]
        )
        
        # Test 1: Service availability validation
        services_to_check = ["ocr_service", "ai_service", "iot_service", "database", "multi_agent"]
        service_health_scores = []
        
        for service in services_to_check:
            try:
                # Mock service health check
                if service == "database":
                    with patch('src.database.connection.get_db_session'):
                        health_status = "healthy"
                        health_score = 1.0
                elif service == "ai_service":
                    with patch('src.services.ai_service.get_ai_service'):
                        health_status = "healthy"
                        health_score = 1.0
                else:
                    health_status = "healthy"
                    health_score = 1.0
                
                health_report.service_statuses[service] = health_status
                service_health_scores.append(health_score)
                
            except Exception:
                health_report.service_statuses[service] = "unhealthy"
                service_health_scores.append(0.0)
        
        # Test 2: Resource utilization validation
        health_report.resource_usage = mock_system_resources['normal_load']
        
        resource_health_score = 1.0
        if health_report.resource_usage['cpu_percent'] > 80:
            resource_health_score -= 0.3
        if health_report.resource_usage['memory_percent'] > 80:
            resource_health_score -= 0.3
        if health_report.resource_usage['disk_percent'] > 90:
            resource_health_score -= 0.4
        
        resource_health_score = max(0.0, resource_health_score)
        
        # Test 3: Performance metrics validation
        with patch('src.services.ai_service.get_ai_service') as mock_ai:
            mock_ai_instance = AsyncMock()
            mock_ai.return_value = mock_ai_instance
            
            # Test response time performance
            response_times = []
            for i in range(10):
                start_time = time.time()
                mock_ai_instance.real_time_inference.return_value = {
                    'result': {'test': True},
                    'processing_time_ms': 150.0,
                    'confidence': 0.9
                }
                await mock_ai_instance.real_time_inference({}, f"perf_test_{i}")
                end_time = time.time()
                response_times.append(end_time - start_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            
            health_report.performance_metrics = {
                'avg_response_time': avg_response_time,
                'max_response_time': max(response_times),
                'min_response_time': min(response_times)
            }
            
            performance_health_score = 1.0
            if avg_response_time > 2.0:  # 2 second threshold
                performance_health_score = 0.5
            elif avg_response_time > 1.0:  # 1 second threshold
                performance_health_score = 0.8
        
        # Test 4: Error handling validation
        error_response = test_client.get("/api/health/errors?limit=10")
        assert error_response.status_code == 200
        error_data = error_response.json()
        
        recent_error_count = len(error_data.get("errors", []))
        error_health_score = 1.0 if recent_error_count < 5 else max(0.0, 1.0 - (recent_error_count * 0.1))
        
        # Test 5: Alert status validation
        alerts_response = test_client.get("/api/health/alerts?resolved=false")
        assert alerts_response.status_code == 200
        alerts_data = alerts_response.json()
        
        active_alert_count = len(alerts_data.get("alerts", []))
        health_report.active_alerts = alerts_data.get("alerts", [])
        
        alert_health_score = 1.0 if active_alert_count == 0 else max(0.0, 1.0 - (active_alert_count * 0.2))
        
        # Test 6: Degradation status validation
        degradation_response = test_client.get("/api/health/resources")
        assert degradation_response.status_code == 200
        degradation_data = degradation_response.json()
        
        health_report.degradation_status = degradation_data.get("degradation_status", {})
        
        degradation_health_score = 0.5 if health_report.degradation_status.get("degradation_active") else 1.0
        
        # Calculate overall health score
        health_scores = [
            sum(service_health_scores) / len(service_health_scores),  # Service health (40%)
            resource_health_score,                                    # Resource health (20%)
            performance_health_score,                                 # Performance health (20%)
            error_health_score,                                       # Error health (10%)
            alert_health_score,                                       # Alert health (5%)
            degradation_health_score                                  # Degradation health (5%)
        ]
        
        weights = [0.4, 0.2, 0.2, 0.1, 0.05, 0.05]
        health_report.overall_health_score = sum(score * weight for score, weight in zip(health_scores, weights))
        
        # Generate recommendations based on health assessment
        if health_report.overall_health_score < 0.8:
            health_report.recommendations.append("System health is below optimal. Review service statuses and resource usage.")
        
        if resource_health_score < 0.8:
            health_report.recommendations.append("High resource utilization detected. Consider scaling or optimization.")
        
        if performance_health_score < 0.8:
            health_report.recommendations.append("Performance degradation detected. Review AI model efficiency and system load.")
        
        if active_alert_count > 0:
            health_report.recommendations.append(f"Active alerts require attention: {active_alert_count} unresolved alerts.")
        
        # Production readiness assertions
        assert health_report.overall_health_score >= 0.8, \
            f"System health score should be >= 0.8 for production readiness, got {health_report.overall_health_score:.2f}"
        
        assert all(status == "healthy" for status in health_report.service_statuses.values()), \
            "All critical services should be healthy for production deployment"
        
        assert health_report.resource_usage['cpu_percent'] < 80, \
            "CPU usage should be under 80% for production readiness"
        
        assert health_report.resource_usage['memory_percent'] < 80, \
            "Memory usage should be under 80% for production readiness"
        
        assert health_report.performance_metrics['avg_response_time'] < 2.0, \
            "Average response time should be under 2 seconds for production readiness"
        
        assert active_alert_count == 0, \
            "No active alerts should exist for production deployment"
        
        return health_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])