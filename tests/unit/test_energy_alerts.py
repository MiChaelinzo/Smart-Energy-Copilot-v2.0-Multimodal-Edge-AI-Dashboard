"""
Tests for the Energy Alerts Service.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from src.services.energy_alerts import (
    EnergyAlertsService,
    AlertType,
    AlertSeverity,
    AlertStatus,
    EnergyAlert
)


class TestEnergyAlertsService:
    """Test cases for EnergyAlertsService."""
    
    @pytest.fixture
    def alerts_service(self):
        """Create alerts service instance."""
        service = EnergyAlertsService()
        return service
    
    @pytest.mark.asyncio
    async def test_service_start(self, alerts_service):
        """Test service starts successfully."""
        result = await alerts_service.start_service()
        assert result is True
        assert alerts_service.is_running is True
    
    @pytest.mark.asyncio
    async def test_service_stop(self, alerts_service):
        """Test service stops successfully."""
        await alerts_service.start_service()
        await alerts_service.stop_service()
        assert alerts_service.is_running is False
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alerts_service):
        """Test creating a new alert."""
        alert = await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            value=100.0,
            threshold=80.0
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.HIGH_CONSUMPTION
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.ACTIVE
        assert alert.title == "Test Alert"
        assert alert.value == 100.0
        assert alert.threshold == 80.0
    
    @pytest.mark.asyncio
    async def test_check_consumption_threshold_high(self, alerts_service):
        """Test high consumption threshold triggers alert."""
        alert = await alerts_service.check_consumption_threshold(
            consumption_kwh=120.0,  # Above default threshold of 100
            device_id="test_device",
            location="Living Room"
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.HIGH_CONSUMPTION
        assert alert.severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_check_consumption_threshold_peak(self, alerts_service):
        """Test peak consumption threshold triggers critical alert."""
        alert = await alerts_service.check_consumption_threshold(
            consumption_kwh=160.0,  # Above peak threshold of 150
            device_id="test_device"
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.PEAK_USAGE
        assert alert.severity == AlertSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_check_consumption_threshold_normal(self, alerts_service):
        """Test normal consumption doesn't trigger alert."""
        alert = await alerts_service.check_consumption_threshold(
            consumption_kwh=50.0  # Below threshold
        )
        
        assert alert is None
    
    @pytest.mark.asyncio
    async def test_check_budget_exceeded(self, alerts_service):
        """Test budget exceeded triggers critical alert."""
        alert, budget_percent = await alerts_service.check_budget_status(
            current_cost=150.0,
            budget_limit=100.0
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.BUDGET_EXCEEDED
        assert alert.severity == AlertSeverity.CRITICAL
        assert budget_percent == 150.0
    
    @pytest.mark.asyncio
    async def test_check_budget_warning(self, alerts_service):
        """Test budget warning triggers warning alert."""
        alert, budget_percent = await alerts_service.check_budget_status(
            current_cost=85.0,  # 85% of budget
            budget_limit=100.0
        )
        
        assert alert is not None
        assert alert.alert_type == AlertType.BUDGET_WARNING
        assert alert.severity == AlertSeverity.WARNING
        assert budget_percent == 85.0
    
    @pytest.mark.asyncio
    async def test_check_budget_normal(self, alerts_service):
        """Test normal budget doesn't trigger alert."""
        alert, budget_percent = await alerts_service.check_budget_status(
            current_cost=50.0,  # 50% of budget
            budget_limit=100.0
        )
        
        assert alert is None
        assert budget_percent == 50.0
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alerts_service):
        """Test acknowledging an alert."""
        # Create an alert first
        alert = await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message"
        )
        
        # Acknowledge it
        result = await alerts_service.acknowledge_alert(alert.alert_id)
        
        assert result is True
        assert alerts_service.alerts[alert.alert_id].status == AlertStatus.ACKNOWLEDGED
        assert alerts_service.alerts[alert.alert_id].acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alerts_service):
        """Test resolving an alert."""
        # Create an alert first
        alert = await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message"
        )
        
        # Resolve it
        result = await alerts_service.resolve_alert(alert.alert_id)
        
        assert result is True
        assert alerts_service.alerts[alert.alert_id].status == AlertStatus.RESOLVED
        assert alerts_service.alerts[alert.alert_id].resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_dismiss_alert(self, alerts_service):
        """Test dismissing an alert."""
        # Create an alert first
        alert = await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message"
        )
        
        # Dismiss it
        result = await alerts_service.dismiss_alert(alert.alert_id)
        
        assert result is True
        assert alerts_service.alerts[alert.alert_id].status == AlertStatus.DISMISSED
    
    @pytest.mark.asyncio
    async def test_get_alerts_filtering(self, alerts_service):
        """Test getting alerts with filters."""
        # Create multiple alerts
        await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test 1",
            message="Message 1"
        )
        await alerts_service.create_alert(
            alert_type=AlertType.BUDGET_WARNING,
            severity=AlertSeverity.CRITICAL,
            title="Test 2",
            message="Message 2"
        )
        
        # Get all active alerts
        active_alerts = alerts_service.get_alerts(status=AlertStatus.ACTIVE)
        assert len(active_alerts) == 2
        
        # Get only warning severity
        warning_alerts = alerts_service.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warning_alerts) == 1
        
        # Get by type
        budget_alerts = alerts_service.get_alerts(alert_type=AlertType.BUDGET_WARNING)
        assert len(budget_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alerts_service):
        """Test getting only active alerts."""
        # Create alerts
        alert1 = await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test 1",
            message="Message 1"
        )
        alert2 = await alerts_service.create_alert(
            alert_type=AlertType.BUDGET_WARNING,
            severity=AlertSeverity.WARNING,
            title="Test 2",
            message="Message 2"
        )
        
        # Resolve one
        await alerts_service.resolve_alert(alert1.alert_id)
        
        # Get active alerts
        active_alerts = alerts_service.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == alert2.alert_id
    
    @pytest.mark.asyncio
    async def test_get_alert_summary(self, alerts_service):
        """Test getting alert summary."""
        # Create alerts with different severities
        await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test 1",
            message="Message 1"
        )
        await alerts_service.create_alert(
            alert_type=AlertType.PEAK_USAGE,
            severity=AlertSeverity.CRITICAL,
            title="Test 2",
            message="Message 2"
        )
        await alerts_service.create_alert(
            alert_type=AlertType.BUDGET_WARNING,
            severity=AlertSeverity.INFO,
            title="Test 3",
            message="Message 3"
        )
        
        summary = alerts_service.get_alert_summary()
        
        assert summary['total_alerts'] == 3
        assert summary['active_alerts'] == 3
        assert summary['severity_breakdown']['critical'] == 1
        assert summary['severity_breakdown']['warning'] == 1
        assert summary['severity_breakdown']['info'] == 1
    
    @pytest.mark.asyncio
    async def test_default_rules_initialized(self, alerts_service):
        """Test default alert rules are initialized."""
        rules = alerts_service.get_rules()
        
        assert len(rules) > 0
        rule_ids = [rule.rule_id for rule in rules]
        assert 'rule_high_consumption' in rule_ids
        assert 'rule_budget_warning' in rule_ids
        assert 'rule_budget_exceeded' in rule_ids
    
    @pytest.mark.asyncio
    async def test_update_rule(self, alerts_service):
        """Test updating an alert rule."""
        result = alerts_service.update_rule(
            'rule_high_consumption',
            {'threshold': 120.0}
        )
        
        assert result is True
        rule = alerts_service.rules['rule_high_consumption']
        assert rule.threshold == 120.0
    
    @pytest.mark.asyncio
    async def test_enable_disable_rule(self, alerts_service):
        """Test enabling and disabling rules."""
        # Disable
        result = alerts_service.disable_rule('rule_high_consumption')
        assert result is True
        assert alerts_service.rules['rule_high_consumption'].enabled is False
        
        # Enable
        result = alerts_service.enable_rule('rule_high_consumption')
        assert result is True
        assert alerts_service.rules['rule_high_consumption'].enabled is True
    
    @pytest.mark.asyncio
    async def test_subscriber_notification(self, alerts_service):
        """Test subscribers are notified of new alerts."""
        received_alerts = []
        
        async def subscriber(alert):
            received_alerts.append(alert)
        
        alerts_service.subscribe(subscriber)
        
        await alerts_service.create_alert(
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message"
        )
        
        assert len(received_alerts) == 1
        assert received_alerts[0].title == "Test"
    
    def test_alert_to_dict(self, alerts_service):
        """Test alert serialization to dictionary."""
        alert = EnergyAlert(
            alert_id="test_id",
            alert_type=AlertType.HIGH_CONSUMPTION,
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now(),
            device_id="device_1",
            location="Living Room",
            value=100.0,
            threshold=80.0,
            metadata={"key": "value"}
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_id'] == "test_id"
        assert alert_dict['alert_type'] == "high_consumption"
        assert alert_dict['severity'] == "warning"
        assert alert_dict['status'] == "active"
        assert alert_dict['device_id'] == "device_1"
    
    def test_calculate_budget_percent(self, alerts_service):
        """Test budget percentage calculation."""
        # Normal calculation
        assert alerts_service.calculate_budget_percent(50.0, 100.0) == 50.0
        assert alerts_service.calculate_budget_percent(100.0, 100.0) == 100.0
        assert alerts_service.calculate_budget_percent(150.0, 100.0) == 150.0
        
        # Edge case: zero budget limit
        assert alerts_service.calculate_budget_percent(50.0, 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
