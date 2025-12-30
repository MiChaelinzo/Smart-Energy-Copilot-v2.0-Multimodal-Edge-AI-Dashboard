"""
Energy Alerts Service - Real-time energy consumption alerts and notifications.

Provides intelligent alerting for energy consumption anomalies, budget thresholds,
peak usage warnings, and device status notifications.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.config.logging import get_logger

logger = get_logger(__name__)


class AlertType(Enum):
    """Types of energy alerts."""
    HIGH_CONSUMPTION = "high_consumption"
    LOW_CONSUMPTION = "low_consumption"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    PEAK_USAGE = "peak_usage"
    DEVICE_ANOMALY = "device_anomaly"
    DEVICE_OFFLINE = "device_offline"
    EFFICIENCY_DROP = "efficiency_drop"
    FORECAST_WARNING = "forecast_warning"
    COST_SPIKE = "cost_spike"
    PATTERN_CHANGE = "pattern_change"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class EnergyAlert:
    """Energy alert representation."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    timestamp: datetime
    device_id: Optional[str]
    location: Optional[str]
    value: Optional[float]
    threshold: Optional[float]
    metadata: Dict[str, Any]
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "location": self.location,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    alert_type: AlertType
    condition: str  # "gt", "lt", "eq", "change"
    threshold: float
    duration_minutes: int  # How long condition must persist
    severity: AlertSeverity
    enabled: bool
    notification_channels: List[str]  # email, push, sms, webhook
    cooldown_minutes: int  # Minimum time between alerts
    created_at: datetime
    last_triggered: Optional[datetime] = None


class EnergyAlertsService:
    """Real-time energy alerts and notifications service."""
    
    def __init__(self):
        self.alerts: Dict[str, EnergyAlert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.subscribers: List[Callable] = []
        self.is_running = False
        self._alert_counter = 0
        
        # Default thresholds
        self.default_thresholds = {
            "high_consumption_kwh": 100.0,
            "budget_warning_percent": 80.0,
            "budget_exceeded_percent": 100.0,
            "peak_usage_kwh": 150.0,
            "efficiency_drop_percent": 15.0,
            "cost_spike_percent": 25.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="rule_high_consumption",
                name="High Consumption Alert",
                alert_type=AlertType.HIGH_CONSUMPTION,
                condition="gt",
                threshold=self.default_thresholds["high_consumption_kwh"],
                duration_minutes=15,
                severity=AlertSeverity.WARNING,
                enabled=True,
                notification_channels=["push", "email"],
                cooldown_minutes=60,
                created_at=datetime.now()
            ),
            AlertRule(
                rule_id="rule_budget_warning",
                name="Budget Warning",
                alert_type=AlertType.BUDGET_WARNING,
                condition="gt",
                threshold=self.default_thresholds["budget_warning_percent"],
                duration_minutes=0,
                severity=AlertSeverity.WARNING,
                enabled=True,
                notification_channels=["push"],
                cooldown_minutes=1440,  # Once per day
                created_at=datetime.now()
            ),
            AlertRule(
                rule_id="rule_budget_exceeded",
                name="Budget Exceeded",
                alert_type=AlertType.BUDGET_EXCEEDED,
                condition="gt",
                threshold=self.default_thresholds["budget_exceeded_percent"],
                duration_minutes=0,
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                notification_channels=["push", "email", "sms"],
                cooldown_minutes=1440,
                created_at=datetime.now()
            ),
            AlertRule(
                rule_id="rule_peak_usage",
                name="Peak Usage Alert",
                alert_type=AlertType.PEAK_USAGE,
                condition="gt",
                threshold=self.default_thresholds["peak_usage_kwh"],
                duration_minutes=5,
                severity=AlertSeverity.INFO,
                enabled=True,
                notification_channels=["push"],
                cooldown_minutes=30,
                created_at=datetime.now()
            ),
            AlertRule(
                rule_id="rule_efficiency_drop",
                name="Efficiency Drop Alert",
                alert_type=AlertType.EFFICIENCY_DROP,
                condition="change",
                threshold=self.default_thresholds["efficiency_drop_percent"],
                duration_minutes=60,
                severity=AlertSeverity.WARNING,
                enabled=True,
                notification_channels=["push", "email"],
                cooldown_minutes=240,
                created_at=datetime.now()
            ),
            AlertRule(
                rule_id="rule_device_offline",
                name="Device Offline Alert",
                alert_type=AlertType.DEVICE_OFFLINE,
                condition="eq",
                threshold=0,
                duration_minutes=10,
                severity=AlertSeverity.WARNING,
                enabled=True,
                notification_channels=["push"],
                cooldown_minutes=30,
                created_at=datetime.now()
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    async def start_service(self) -> bool:
        """Start the energy alerts service."""
        logger.info("Starting Energy Alerts Service...")
        
        try:
            self.is_running = True
            logger.info("Energy Alerts Service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Energy Alerts Service: {e}")
            return False
    
    async def stop_service(self):
        """Stop the energy alerts service."""
        self.is_running = False
        logger.info("Energy Alerts Service stopped")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._alert_counter}"
    
    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        device_id: Optional[str] = None,
        location: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnergyAlert:
        """Create a new energy alert."""
        try:
            alert = EnergyAlert(
                alert_id=self._generate_alert_id(),
                alert_type=alert_type,
                severity=severity,
                status=AlertStatus.ACTIVE,
                title=title,
                message=message,
                timestamp=datetime.now(),
                device_id=device_id,
                location=location,
                value=value,
                threshold=threshold,
                metadata=metadata or {}
            )
            
            self.alerts[alert.alert_id] = alert
            
            # Notify subscribers
            await self._notify_subscribers(alert)
            
            logger.info(f"Created alert: {alert.title} [{severity.value}]")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    async def check_consumption_threshold(
        self,
        consumption_kwh: float,
        device_id: Optional[str] = None,
        location: Optional[str] = None
    ) -> Optional[EnergyAlert]:
        """Check if consumption exceeds threshold and create alert if needed."""
        high_threshold = self.default_thresholds["high_consumption_kwh"]
        peak_threshold = self.default_thresholds["peak_usage_kwh"]
        
        if consumption_kwh >= peak_threshold:
            return await self.create_alert(
                alert_type=AlertType.PEAK_USAGE,
                severity=AlertSeverity.CRITICAL,
                title="Peak Energy Usage Detected",
                message=f"Current consumption ({consumption_kwh:.1f} kWh) has exceeded peak threshold ({peak_threshold} kWh)",
                device_id=device_id,
                location=location,
                value=consumption_kwh,
                threshold=peak_threshold,
                metadata={"exceeds_peak": True}
            )
        elif consumption_kwh >= high_threshold:
            return await self.create_alert(
                alert_type=AlertType.HIGH_CONSUMPTION,
                severity=AlertSeverity.WARNING,
                title="High Energy Consumption",
                message=f"Current consumption ({consumption_kwh:.1f} kWh) exceeds normal threshold ({high_threshold} kWh)",
                device_id=device_id,
                location=location,
                value=consumption_kwh,
                threshold=high_threshold,
                metadata={"exceeds_normal": True}
            )
        
        return None
    
    async def check_budget_status(
        self,
        current_cost: float,
        budget_limit: float
    ) -> Optional[EnergyAlert]:
        """Check budget status and create alert if threshold exceeded."""
        budget_percent = (current_cost / budget_limit) * 100 if budget_limit > 0 else 0
        
        warning_threshold = self.default_thresholds["budget_warning_percent"]
        exceeded_threshold = self.default_thresholds["budget_exceeded_percent"]
        
        if budget_percent >= exceeded_threshold:
            return await self.create_alert(
                alert_type=AlertType.BUDGET_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                title="Energy Budget Exceeded",
                message=f"Your energy spending (${current_cost:.2f}) has exceeded your budget (${budget_limit:.2f})",
                value=current_cost,
                threshold=budget_limit,
                metadata={"budget_percent": budget_percent, "overage": current_cost - budget_limit}
            )
        elif budget_percent >= warning_threshold:
            return await self.create_alert(
                alert_type=AlertType.BUDGET_WARNING,
                severity=AlertSeverity.WARNING,
                title="Budget Warning",
                message=f"You've used {budget_percent:.1f}% of your energy budget (${current_cost:.2f} of ${budget_limit:.2f})",
                value=current_cost,
                threshold=budget_limit,
                metadata={"budget_percent": budget_percent, "remaining": budget_limit - current_cost}
            )
        
        return None
    
    async def check_device_status(
        self,
        device_id: str,
        device_name: str,
        is_online: bool,
        location: Optional[str] = None
    ) -> Optional[EnergyAlert]:
        """Check device status and create alert if offline."""
        if not is_online:
            return await self.create_alert(
                alert_type=AlertType.DEVICE_OFFLINE,
                severity=AlertSeverity.WARNING,
                title=f"Device Offline: {device_name}",
                message=f"The device '{device_name}' is currently offline and may need attention",
                device_id=device_id,
                location=location,
                metadata={"device_name": device_name, "offline_since": datetime.now().isoformat()}
            )
        
        return None
    
    async def check_cost_spike(
        self,
        current_rate: float,
        average_rate: float
    ) -> Optional[EnergyAlert]:
        """Check for cost spikes compared to average."""
        if average_rate <= 0:
            return None
        
        spike_percent = ((current_rate - average_rate) / average_rate) * 100
        spike_threshold = self.default_thresholds["cost_spike_percent"]
        
        if spike_percent >= spike_threshold:
            return await self.create_alert(
                alert_type=AlertType.COST_SPIKE,
                severity=AlertSeverity.WARNING,
                title="Cost Spike Detected",
                message=f"Current energy rate (${current_rate:.4f}/kWh) is {spike_percent:.1f}% higher than average (${average_rate:.4f}/kWh)",
                value=current_rate,
                threshold=average_rate,
                metadata={"spike_percent": spike_percent}
            )
        
        return None
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        if alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    async def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            alert.status = AlertStatus.DISMISSED
            logger.info(f"Alert dismissed: {alert_id}")
            return True
        
        return False
    
    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        device_id: Optional[str] = None,
        limit: int = 50
    ) -> List[EnergyAlert]:
        """Get alerts with optional filtering."""
        alerts = list(self.alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if device_id:
            alerts = [a for a in alerts if a.device_id == device_id]
        
        # Sort by timestamp (newest first) and limit
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
        
        return alerts
    
    def get_active_alerts(self) -> List[EnergyAlert]:
        """Get all active alerts."""
        return self.get_alerts(status=AlertStatus.ACTIVE)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts."""
        all_alerts = list(self.alerts.values())
        
        active_count = len([a for a in all_alerts if a.status == AlertStatus.ACTIVE])
        acknowledged_count = len([a for a in all_alerts if a.status == AlertStatus.ACKNOWLEDGED])
        resolved_count = len([a for a in all_alerts if a.status == AlertStatus.RESOLVED])
        
        severity_counts = {
            "critical": len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL and a.status == AlertStatus.ACTIVE]),
            "warning": len([a for a in all_alerts if a.severity == AlertSeverity.WARNING and a.status == AlertStatus.ACTIVE]),
            "info": len([a for a in all_alerts if a.severity == AlertSeverity.INFO and a.status == AlertStatus.ACTIVE])
        }
        
        type_counts = {}
        for alert in all_alerts:
            if alert.status == AlertStatus.ACTIVE:
                type_name = alert.alert_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_alerts": len(all_alerts),
            "active_alerts": active_count,
            "acknowledged_alerts": acknowledged_count,
            "resolved_alerts": resolved_count,
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "timestamp": datetime.now().isoformat()
        }
    
    def subscribe(self, callback: Callable):
        """Subscribe to alert notifications."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from alert notifications."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _notify_subscribers(self, alert: EnergyAlert):
        """Notify all subscribers of a new alert."""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(alert)
                else:
                    subscriber(alert)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self.rules.values())
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert rule."""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"Updated alert rule: {rule_id}")
        return True
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        return self.update_rule(rule_id, {"enabled": True})
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        return self.update_rule(rule_id, {"enabled": False})


# Global energy alerts service instance
energy_alerts_service = EnergyAlertsService()
