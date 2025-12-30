"""
Energy Alerts API endpoints.

Provides REST API for managing energy alerts, rules, and notifications.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from src.services.energy_alerts import (
    energy_alerts_service, AlertType, AlertSeverity, AlertStatus
)
from src.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# Request/Response Models
class CreateAlertRequest(BaseModel):
    """Request model for creating an alert."""
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(default="warning", description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    device_id: Optional[str] = None
    location: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    """Response model for an alert."""
    alert_id: str
    alert_type: str
    severity: str
    status: str
    title: str
    message: str
    timestamp: str
    device_id: Optional[str]
    location: Optional[str]
    value: Optional[float]
    threshold: Optional[float]
    metadata: Dict[str, Any]
    acknowledged_at: Optional[str]
    resolved_at: Optional[str]


class AlertSummaryResponse(BaseModel):
    """Response model for alert summary."""
    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    severity_breakdown: Dict[str, int]
    type_breakdown: Dict[str, int]
    timestamp: str


class CheckConsumptionRequest(BaseModel):
    """Request for checking consumption threshold."""
    consumption_kwh: float
    device_id: Optional[str] = None
    location: Optional[str] = None


class CheckBudgetRequest(BaseModel):
    """Request for checking budget status."""
    current_cost: float
    budget_limit: float


class UpdateRuleRequest(BaseModel):
    """Request for updating an alert rule."""
    threshold: Optional[float] = None
    severity: Optional[str] = None
    enabled: Optional[bool] = None
    cooldown_minutes: Optional[int] = None
    notification_channels: Optional[List[str]] = None


# API Endpoints

@router.get("/", response_model=Dict[str, Any])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum alerts to return")
):
    """Get all alerts with optional filtering."""
    try:
        # Convert string filters to enums
        status_enum = AlertStatus(status) if status else None
        type_enum = AlertType(alert_type) if alert_type else None
        severity_enum = AlertSeverity(severity) if severity else None
        
        alerts = energy_alerts_service.get_alerts(
            status=status_enum,
            alert_type=type_enum,
            severity=severity_enum,
            device_id=device_id,
            limit=limit
        )
        
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts),
            "filters": {
                "status": status,
                "alert_type": alert_type,
                "severity": severity,
                "device_id": device_id,
                "limit": limit
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.get("/active", response_model=Dict[str, Any])
async def get_active_alerts():
    """Get all active alerts."""
    try:
        alerts = energy_alerts_service.get_active_alerts()
        
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active alerts")


@router.get("/summary", response_model=AlertSummaryResponse)
async def get_alert_summary():
    """Get summary of all alerts."""
    try:
        summary = energy_alerts_service.get_alert_summary()
        return AlertSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert summary")


@router.post("/", response_model=Dict[str, Any])
async def create_alert(request: CreateAlertRequest):
    """Create a new alert manually."""
    try:
        alert_type = AlertType(request.alert_type)
        severity = AlertSeverity(request.severity)
        
        alert = await energy_alerts_service.create_alert(
            alert_type=alert_type,
            severity=severity,
            title=request.title,
            message=request.message,
            device_id=request.device_id,
            location=request.location,
            value=request.value,
            threshold=request.threshold,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "alert": alert.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid alert type or severity: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert")


@router.post("/check/consumption", response_model=Dict[str, Any])
async def check_consumption_threshold(request: CheckConsumptionRequest):
    """Check consumption against threshold and create alert if exceeded."""
    try:
        alert = await energy_alerts_service.check_consumption_threshold(
            consumption_kwh=request.consumption_kwh,
            device_id=request.device_id,
            location=request.location
        )
        
        return {
            "alert_created": alert is not None,
            "alert": alert.to_dict() if alert else None,
            "consumption_kwh": request.consumption_kwh
        }
    except Exception as e:
        logger.error(f"Error checking consumption threshold: {e}")
        raise HTTPException(status_code=500, detail="Failed to check consumption threshold")


@router.post("/check/budget", response_model=Dict[str, Any])
async def check_budget_status(request: CheckBudgetRequest):
    """Check budget status and create alert if threshold exceeded."""
    try:
        alert, budget_percent = await energy_alerts_service.check_budget_status(
            current_cost=request.current_cost,
            budget_limit=request.budget_limit
        )
        
        return {
            "alert_created": alert is not None,
            "alert": alert.to_dict() if alert else None,
            "current_cost": request.current_cost,
            "budget_limit": request.budget_limit,
            "budget_percent": budget_percent
        }
    except Exception as e:
        logger.error(f"Error checking budget status: {e}")
        raise HTTPException(status_code=500, detail="Failed to check budget status")


@router.get("/{alert_id}", response_model=Dict[str, Any])
async def get_alert(alert_id: str):
    """Get a specific alert by ID."""
    try:
        if alert_id not in energy_alerts_service.alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert = energy_alerts_service.alerts[alert_id]
        return {"alert": alert.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert")


@router.post("/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        success = await energy_alerts_service.acknowledge_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or already processed")
        
        return {
            "success": True,
            "message": "Alert acknowledged successfully",
            "alert_id": alert_id,
            "acknowledged_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    try:
        success = await energy_alerts_service.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or already resolved")
        
        return {
            "success": True,
            "message": "Alert resolved successfully",
            "alert_id": alert_id,
            "resolved_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.post("/{alert_id}/dismiss", response_model=Dict[str, Any])
async def dismiss_alert(alert_id: str):
    """Dismiss an alert."""
    try:
        success = await energy_alerts_service.dismiss_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or already dismissed")
        
        return {
            "success": True,
            "message": "Alert dismissed successfully",
            "alert_id": alert_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to dismiss alert")


# Alert Rules Endpoints

@router.get("/rules/", response_model=Dict[str, Any])
async def get_alert_rules():
    """Get all alert rules."""
    try:
        rules = energy_alerts_service.get_rules()
        
        return {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "alert_type": rule.alert_type.value,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "duration_minutes": rule.duration_minutes,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled,
                    "notification_channels": rule.notification_channels,
                    "cooldown_minutes": rule.cooldown_minutes,
                    "created_at": rule.created_at.isoformat(),
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule in rules
            ],
            "count": len(rules)
        }
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert rules")


@router.put("/rules/{rule_id}", response_model=Dict[str, Any])
async def update_alert_rule(rule_id: str, request: UpdateRuleRequest):
    """Update an alert rule."""
    try:
        updates = {}
        if request.threshold is not None:
            updates["threshold"] = request.threshold
        if request.severity is not None:
            updates["severity"] = AlertSeverity(request.severity)
        if request.enabled is not None:
            updates["enabled"] = request.enabled
        if request.cooldown_minutes is not None:
            updates["cooldown_minutes"] = request.cooldown_minutes
        if request.notification_channels is not None:
            updates["notification_channels"] = request.notification_channels
        
        success = energy_alerts_service.update_rule(rule_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        return {
            "success": True,
            "message": "Rule updated successfully",
            "rule_id": rule_id
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update alert rule")


@router.post("/rules/{rule_id}/enable", response_model=Dict[str, Any])
async def enable_alert_rule(rule_id: str):
    """Enable an alert rule."""
    try:
        success = energy_alerts_service.enable_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        return {
            "success": True,
            "message": "Rule enabled successfully",
            "rule_id": rule_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable alert rule")


@router.post("/rules/{rule_id}/disable", response_model=Dict[str, Any])
async def disable_alert_rule(rule_id: str):
    """Disable an alert rule."""
    try:
        success = energy_alerts_service.disable_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        return {
            "success": True,
            "message": "Rule disabled successfully",
            "rule_id": rule_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable alert rule")
