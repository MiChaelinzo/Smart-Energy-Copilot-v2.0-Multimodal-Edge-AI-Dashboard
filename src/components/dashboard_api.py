"""
Web Dashboard Backend API

This module provides REST API endpoints for the energy dashboard,
including energy data retrieval, real-time updates, and recommendation management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import get_db_session
from ..models.energy_consumption import EnergyConsumption, EnergyConsumptionORM
from ..models.sensor_reading import SensorReading, SensorReadingORM
from ..models.recommendation import OptimizationRecommendation, OptimizationRecommendationORM
from ..services.recommendation_engine import get_recommendation_engine
from ..services.ai_service import get_ai_service
from ..services.iot_integration import IoTIntegrationService
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
security = HTTPBearer(auto_error=False)
settings = get_settings()


# Request/Response Models
class EnergyDataRequest(BaseModel):
    """Request model for energy data queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    source: Optional[str] = None


class EnergyDataResponse(BaseModel):
    """Response model for energy data."""
    data: List[Dict[str, Any]]
    total_count: int
    time_range: Dict[str, datetime]
    summary: Dict[str, float]


class RecommendationRequest(BaseModel):
    """Request model for recommendation queries."""
    status: Optional[str] = None
    type: Optional[str] = None
    priority: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[Dict[str, Any]]
    total_count: int
    summary: Dict[str, Any]


class RecommendationUpdateRequest(BaseModel):
    """Request model for updating recommendation status."""
    status: str = Field(..., regex="^(pending|implemented|dismissed)$")
    actual_savings: Optional[Dict[str, float]] = None
    notes: Optional[str] = None


class DashboardConfigRequest(BaseModel):
    """Request model for dashboard configuration."""
    layout: Dict[str, Any]
    preferences: Dict[str, Any]
    refresh_interval: int = Field(default=30, ge=5, le=300)


class UserSession(BaseModel):
    """User session model."""
    user_id: str
    session_id: str
    created_at: datetime
    last_active: datetime
    preferences: Dict[str, Any]


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_to_user(self, user_id: str, message: str):
        """Send a message to all connections for a specific user."""
        if user_id in self.user_connections:
            disconnected = []
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection, user_id)


# Global connection manager
manager = ConnectionManager()


# Authentication dependency (simplified for development)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user from authentication token."""
    if not credentials:
        return None  # Allow anonymous access for development
    
    # In production, this would validate the JWT token
    # For development, we'll use a simple token format
    try:
        # Simple token format: "user_<user_id>"
        if credentials.credentials.startswith("user_"):
            return credentials.credentials[5:]  # Extract user_id
        return None
    except Exception:
        return None


# API Endpoints

@router.get("/energy-data", response_model=EnergyDataResponse)
async def get_energy_data(
    request: EnergyDataRequest = Depends(),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Retrieve energy consumption data for dashboard display.
    
    **Validates: Requirements 5.1**
    """
    try:
        async with get_db_session() as session:
            # Build query
            query = session.query(EnergyConsumptionORM)
            
            # Apply filters
            if request.start_date:
                query = query.filter(EnergyConsumptionORM.timestamp >= request.start_date)
            if request.end_date:
                query = query.filter(EnergyConsumptionORM.timestamp <= request.end_date)
            if request.source:
                query = query.filter(EnergyConsumptionORM.source == request.source)
            
            # Get total count
            total_count = query.count()
            
            # Apply limit and get results
            results = query.order_by(EnergyConsumptionORM.timestamp.desc()).limit(request.limit).all()
            
            # Convert to response format
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'timestamp': result.timestamp,
                    'source': result.source,
                    'consumption_kwh': result.consumption_kwh,
                    'cost_usd': result.cost_usd,
                    'billing_period_start': result.billing_period_start,
                    'billing_period_end': result.billing_period_end,
                    'confidence_score': result.confidence_score
                })
            
            # Calculate summary statistics
            if data:
                total_consumption = sum(item['consumption_kwh'] for item in data)
                total_cost = sum(item['cost_usd'] for item in data)
                avg_confidence = sum(item['confidence_score'] for item in data) / len(data)
                
                time_range = {
                    'start': min(item['timestamp'] for item in data),
                    'end': max(item['timestamp'] for item in data)
                }
                
                summary = {
                    'total_consumption_kwh': total_consumption,
                    'total_cost_usd': total_cost,
                    'average_confidence': avg_confidence,
                    'records_count': len(data)
                }
            else:
                time_range = {
                    'start': request.start_date or datetime.now() - timedelta(days=30),
                    'end': request.end_date or datetime.now()
                }
                summary = {
                    'total_consumption_kwh': 0.0,
                    'total_cost_usd': 0.0,
                    'average_confidence': 0.0,
                    'records_count': 0
                }
            
            return EnergyDataResponse(
                data=data,
                total_count=total_count,
                time_range=time_range,
                summary=summary
            )
    
    except Exception as e:
        logger.error(f"Error retrieving energy data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve energy data")


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest = Depends(),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Retrieve optimization recommendations for dashboard display.
    
    **Validates: Requirements 5.2**
    """
    try:
        async with get_db_session() as session:
            # Build query
            query = session.query(OptimizationRecommendationORM)
            
            # Apply filters
            if request.status:
                query = query.filter(OptimizationRecommendationORM.status == request.status)
            if request.type:
                query = query.filter(OptimizationRecommendationORM.type == request.type)
            if request.priority:
                query = query.filter(OptimizationRecommendationORM.priority == request.priority)
            
            # Get total count
            total_count = query.count()
            
            # Apply limit and get results
            results = query.order_by(OptimizationRecommendationORM.created_at.desc()).limit(request.limit).all()
            
            # Convert to response format
            recommendations = []
            for result in results:
                recommendations.append({
                    'id': result.id,
                    'type': result.type,
                    'priority': result.priority,
                    'title': result.title,
                    'description': result.description,
                    'implementation_steps': result.implementation_steps,
                    'estimated_savings': result.estimated_savings,
                    'difficulty': result.difficulty,
                    'agent_source': result.agent_source,
                    'confidence': result.confidence,
                    'created_at': result.created_at,
                    'status': result.status
                })
            
            # Calculate summary statistics
            if recommendations:
                total_savings = sum(
                    rec['estimated_savings'].get('annual_cost_usd', 0) 
                    for rec in recommendations 
                    if rec['estimated_savings']
                )
                avg_confidence = sum(rec['confidence'] for rec in recommendations) / len(recommendations)
                
                status_counts = {}
                priority_counts = {}
                type_counts = {}
                
                for rec in recommendations:
                    status_counts[rec['status']] = status_counts.get(rec['status'], 0) + 1
                    priority_counts[rec['priority']] = priority_counts.get(rec['priority'], 0) + 1
                    type_counts[rec['type']] = type_counts.get(rec['type'], 0) + 1
                
                summary = {
                    'total_estimated_savings': total_savings,
                    'average_confidence': avg_confidence,
                    'status_distribution': status_counts,
                    'priority_distribution': priority_counts,
                    'type_distribution': type_counts,
                    'recommendations_count': len(recommendations)
                }
            else:
                summary = {
                    'total_estimated_savings': 0.0,
                    'average_confidence': 0.0,
                    'status_distribution': {},
                    'priority_distribution': {},
                    'type_distribution': {},
                    'recommendations_count': 0
                }
            
            return RecommendationResponse(
                recommendations=recommendations,
                total_count=total_count,
                summary=summary
            )
    
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")


@router.put("/recommendations/{recommendation_id}")
async def update_recommendation(
    recommendation_id: str,
    request: RecommendationUpdateRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Update recommendation status and track implementation progress.
    
    **Validates: Requirements 5.4**
    """
    try:
        async with get_db_session() as session:
            # Find the recommendation
            recommendation = session.query(OptimizationRecommendationORM).filter(
                OptimizationRecommendationORM.id == recommendation_id
            ).first()
            
            if not recommendation:
                raise HTTPException(status_code=404, detail="Recommendation not found")
            
            # Update fields
            recommendation.status = request.status
            if request.actual_savings:
                recommendation.actual_savings = request.actual_savings
            if request.notes:
                recommendation.notes = request.notes
            
            session.commit()
            
            # Broadcast update to connected clients
            update_message = {
                'type': 'recommendation_update',
                'recommendation_id': recommendation_id,
                'status': request.status,
                'timestamp': datetime.now().isoformat()
            }
            await manager.broadcast(json.dumps(update_message))
            
            return {"message": "Recommendation updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating recommendation: {e}")
        raise HTTPException(status_code=500, detail="Failed to update recommendation")


@router.get("/config")
async def get_dashboard_config(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Get dashboard configuration and user preferences.
    
    **Validates: Requirements 5.4**
    """
    try:
        # For development, return default configuration
        # In production, this would be stored per user
        default_config = {
            'layout': {
                'widgets': [
                    {'type': 'energy_overview', 'position': {'x': 0, 'y': 0, 'w': 6, 'h': 4}},
                    {'type': 'recommendations', 'position': {'x': 6, 'y': 0, 'w': 6, 'h': 4}},
                    {'type': 'consumption_chart', 'position': {'x': 0, 'y': 4, 'w': 12, 'h': 6}},
                    {'type': 'device_status', 'position': {'x': 0, 'y': 10, 'w': 6, 'h': 4}},
                    {'type': 'cost_analysis', 'position': {'x': 6, 'y': 10, 'w': 6, 'h': 4}}
                ]
            },
            'preferences': {
                'theme': 'light',
                'currency': 'USD',
                'units': 'metric',
                'refresh_interval': 30,
                'notifications_enabled': True,
                'auto_refresh': True
            },
            'refresh_interval': 30
        }
        
        return default_config
    
    except Exception as e:
        logger.error(f"Error retrieving dashboard config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard configuration")


@router.put("/config")
async def update_dashboard_config(
    request: DashboardConfigRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Update dashboard configuration and user preferences.
    
    **Validates: Requirements 5.4**
    """
    try:
        # For development, just validate and return success
        # In production, this would be stored per user in database
        
        # Validate layout structure
        if 'widgets' not in request.layout:
            raise HTTPException(status_code=400, detail="Layout must contain widgets array")
        
        # Validate refresh interval
        if not (5 <= request.refresh_interval <= 300):
            raise HTTPException(status_code=400, detail="Refresh interval must be between 5 and 300 seconds")
        
        logger.info(f"Dashboard configuration updated for user: {current_user}")
        
        return {"message": "Dashboard configuration updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dashboard config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update dashboard configuration")


@router.get("/session")
async def get_user_session(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Get current user session information.
    """
    try:
        if not current_user:
            return {"authenticated": False}
        
        # For development, return mock session data
        session_data = {
            'authenticated': True,
            'user_id': current_user,
            'session_id': f"session_{current_user}_{datetime.now().timestamp()}",
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'preferences': {
                'theme': 'light',
                'notifications': True
            }
        }
        
        return session_data
    
    except Exception as e:
        logger.error(f"Error retrieving user session: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user session")


@router.post("/session/refresh")
async def refresh_user_session(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Refresh user session and update last active timestamp.
    """
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # For development, just return success
        # In production, this would update session in database
        
        return {
            "message": "Session refreshed successfully",
            "last_active": datetime.now()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing user session: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh user session")


# WebSocket endpoint for real-time updates
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time dashboard updates.
    
    **Validates: Requirements 5.4**
    """
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get('type') == 'ping':
                await manager.send_personal_message(
                    json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get('type') == 'subscribe':
                # Handle subscription to specific data streams
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'subscription_confirmed',
                        'stream': message_data.get('stream'),
                        'timestamp': datetime.now().isoformat()
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)


# Background task for sending real-time updates
async def send_real_time_updates():
    """
    Background task that sends periodic updates to connected WebSocket clients.
    """
    while True:
        try:
            # Get latest energy data
            async with get_db_session() as session:
                latest_consumption = session.query(EnergyConsumptionORM).order_by(
                    EnergyConsumptionORM.timestamp.desc()
                ).first()
                
                latest_recommendations = session.query(OptimizationRecommendationORM).filter(
                    OptimizationRecommendationORM.status == 'pending'
                ).order_by(OptimizationRecommendationORM.created_at.desc()).limit(5).all()
            
            # Prepare update message
            update_data = {
                'type': 'real_time_update',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'latest_consumption': {
                        'consumption_kwh': latest_consumption.consumption_kwh if latest_consumption else 0,
                        'cost_usd': latest_consumption.cost_usd if latest_consumption else 0,
                        'timestamp': latest_consumption.timestamp.isoformat() if latest_consumption else None
                    },
                    'pending_recommendations_count': len(latest_recommendations),
                    'system_status': 'online'
                }
            }
            
            # Broadcast to all connected clients
            await manager.broadcast(json.dumps(update_data))
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# Start background task when module is imported
asyncio.create_task(send_real_time_updates())