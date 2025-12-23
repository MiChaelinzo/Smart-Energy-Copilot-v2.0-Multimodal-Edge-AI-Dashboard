"""
AI API endpoints for ERNIE model integration.

Provides REST API endpoints for energy pattern analysis, data fusion,
and real-time AI inference capabilities.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging

from ..services.ai_service import get_ai_service
from ..models.energy_consumption import EnergyConsumption
from ..models.sensor_reading import SensorReading

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI"])


class PatternAnalysisRequest(BaseModel):
    """Request model for pattern analysis."""
    consumption_data: List[Dict[str, Any]]
    sensor_data: Optional[List[Dict[str, Any]]] = None
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")


class PatternAnalysisResponse(BaseModel):
    """Response model for pattern analysis."""
    patterns: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    processing_time_ms: float
    confidence_score: float


class RealTimeInferenceRequest(BaseModel):
    """Request model for real-time inference."""
    data: Dict[str, Any]
    inference_type: str = Field(default="pattern_analysis", description="Type of inference")


class RealTimeInferenceResponse(BaseModel):
    """Response model for real-time inference."""
    result: Dict[str, Any]
    processing_time_ms: float
    confidence: float
    timestamp: datetime


class DataFusionRequest(BaseModel):
    """Request model for data fusion."""
    utility_data: List[Dict[str, Any]]
    sensor_data: List[Dict[str, Any]]
    external_data: Optional[Dict[str, Any]] = None


class DataFusionResponse(BaseModel):
    """Response model for data fusion."""
    fused_data: Dict[str, Any]
    data_quality_score: float
    processing_time_ms: float


@router.post("/analyze-patterns", response_model=PatternAnalysisResponse)
async def analyze_energy_patterns(request: PatternAnalysisRequest):
    """
    Analyze energy consumption patterns using ERNIE AI model.
    
    This endpoint processes energy consumption data to identify patterns,
    trends, and anomalies using the fine-tuned ERNIE model.
    """
    try:
        ai_service = await get_ai_service()
        
        # Convert request data to model objects
        consumption_data = []
        for item in request.consumption_data:
            consumption = EnergyConsumption(**item)
            consumption_data.append(consumption)
        
        sensor_data = []
        if request.sensor_data:
            for item in request.sensor_data:
                sensor = SensorReading(**item)
                sensor_data.append(sensor)
        
        # Perform batch inference
        result = await ai_service.batch_inference(
            consumption_data + sensor_data,
            inference_type="comprehensive_analysis"
        )
        
        return PatternAnalysisResponse(
            patterns=result['results'].get('patterns', []),
            insights=result['results'].get('fusion_data', {}).get('combined_insights', []),
            processing_time_ms=result['processing_time_ms'],
            confidence_score=result['summary'].get('data_quality_score', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@router.post("/real-time-inference", response_model=RealTimeInferenceResponse)
async def real_time_inference(request: RealTimeInferenceRequest):
    """
    Perform real-time AI inference on energy data.
    
    This endpoint provides sub-second inference for real-time energy
    analysis and pattern detection.
    """
    try:
        ai_service = await get_ai_service()
        
        # Determine data type and create appropriate object
        if request.inference_type == "pattern_analysis":
            if 'consumption_kwh' in request.data:
                data_obj = EnergyConsumption(**request.data)
            elif 'sensor_id' in request.data:
                data_obj = SensorReading(**request.data)
            else:
                raise ValueError("Invalid data format for pattern analysis")
        else:
            data_obj = request.data
        
        # Perform real-time inference
        result = await ai_service.real_time_inference(
            data_obj,
            inference_type=request.inference_type
        )
        
        return RealTimeInferenceResponse(
            result=result['result'],
            processing_time_ms=result['processing_time_ms'],
            confidence=result['confidence'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error in real-time inference: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time inference failed: {str(e)}")


@router.post("/fuse-data", response_model=DataFusionResponse)
async def fuse_multi_source_data(request: DataFusionRequest):
    """
    Combine data from multiple sources for comprehensive analysis.
    
    This endpoint fuses utility bills, IoT sensor data, and external
    sources to provide unified energy insights.
    """
    try:
        ai_service = await get_ai_service()
        
        # Convert request data to model objects
        utility_data = [EnergyConsumption(**item) for item in request.utility_data]
        sensor_data = [SensorReading(**item) for item in request.sensor_data]
        
        # Perform data fusion
        start_time = datetime.now()
        fused_result = await ai_service.fusion_engine.fuse_multi_source_data(
            utility_data,
            sensor_data,
            request.external_data
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return DataFusionResponse(
            fused_data=fused_result,
            data_quality_score=fused_result['data_quality_score'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in data fusion: {e}")
        raise HTTPException(status_code=500, detail=f"Data fusion failed: {str(e)}")


@router.get("/model-status")
async def get_model_status():
    """
    Get the current status of the ERNIE AI model.
    
    Returns information about model loading status, memory usage,
    and performance metrics.
    """
    try:
        ai_service = await get_ai_service()
        
        return {
            "model_loaded": ai_service.model_manager.is_loaded(),
            "model_path": ai_service.model_manager.model_path,
            "device": str(ai_service.model_manager.device),
            "status": "ready" if ai_service.model_manager.is_loaded() else "loading",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/analyze-text")
async def analyze_energy_text(text: str):
    """
    Analyze energy-related text using the ERNIE model.
    
    This endpoint processes natural language text to extract
    energy-related insights and concepts.
    """
    try:
        ai_service = await get_ai_service()
        
        # Perform text analysis
        result = await ai_service.real_time_inference(
            text,
            inference_type="text_analysis"
        )
        
        return {
            "analysis": result['result'],
            "processing_time_ms": result['processing_time_ms'],
            "confidence": result['confidence'],
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")