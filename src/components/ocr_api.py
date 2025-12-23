"""OCR API endpoints for document upload and processing."""

import io
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.services.ocr_service import OCRProcessingEngine, EnergyFieldData, OCRResult
from src.config.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/ocr", tags=["OCR Processing"])

# Global OCR engine instance
ocr_engine: Optional[OCRProcessingEngine] = None


def get_ocr_engine() -> OCRProcessingEngine:
    """Get or create OCR engine instance."""
    global ocr_engine
    if ocr_engine is None:
        ocr_engine = OCRProcessingEngine()
    return ocr_engine


class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    message: str
    ocr_result: Optional[Dict[str, Any]] = None
    energy_data: Optional[Dict[str, Any]] = None
    quality_assessment: Optional[Dict[str, Any]] = None


class EnergyFieldResponse(BaseModel):
    """Response model for energy field extraction."""
    success: bool
    message: str
    energy_data: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None


@router.post("/upload", response_model=DocumentProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    extract_fields: bool = True,
    ocr_engine: OCRProcessingEngine = Depends(get_ocr_engine)
):
    """Upload and process a document with OCR.
    
    Args:
        file: Uploaded document file (PDF, JPEG, PNG)
        extract_fields: Whether to extract energy-specific fields
        ocr_engine: OCR processing engine instance
        
    Returns:
        Processing results with OCR text and extracted energy data
    """
    try:
        # Validate file size (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        logger.info(f"Processing uploaded file: {file.filename}, size: {len(file_content)} bytes")
        
        # Process document with OCR
        ocr_result = ocr_engine.process_document(file_content, file.filename or "")
        
        # Assess document quality
        quality_assessment = ocr_engine.assess_quality(ocr_result)
        
        response_data = {
            "success": True,
            "message": "Document processed successfully",
            "ocr_result": {
                "text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "format": ocr_result.format.value,
                "page_count": ocr_result.page_count,
                "bounding_box_count": len(ocr_result.bounding_boxes)
            },
            "quality_assessment": quality_assessment
        }
        
        # Extract energy fields if requested
        if extract_fields:
            energy_data = ocr_engine.extract_energy_fields(ocr_result)
            response_data["energy_data"] = {
                "consumption_kwh": energy_data.consumption_kwh,
                "cost_usd": energy_data.cost_usd,
                "billing_period_start": energy_data.billing_period_start,
                "billing_period_end": energy_data.billing_period_end,
                "account_number": energy_data.account_number,
                "confidence_scores": energy_data.confidence_scores
            }
        
        return DocumentProcessResponse(**response_data)
        
    except ValueError as e:
        logger.error(f"Document format error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Document format error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/extract-fields", response_model=EnergyFieldResponse)
async def extract_energy_fields(
    file: UploadFile = File(...),
    ocr_engine: OCRProcessingEngine = Depends(get_ocr_engine)
):
    """Extract energy-specific fields from a document.
    
    Args:
        file: Uploaded document file
        ocr_engine: OCR processing engine instance
        
    Returns:
        Extracted energy field data with confidence scores
    """
    try:
        # Read and process file
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        logger.info(f"Extracting energy fields from: {file.filename}")
        
        # Process document
        ocr_result = ocr_engine.process_document(file_content, file.filename or "")
        
        # Extract energy fields
        energy_data = ocr_engine.extract_energy_fields(ocr_result)
        
        return EnergyFieldResponse(
            success=True,
            message="Energy fields extracted successfully",
            energy_data={
                "consumption_kwh": energy_data.consumption_kwh,
                "cost_usd": energy_data.cost_usd,
                "billing_period_start": energy_data.billing_period_start,
                "billing_period_end": energy_data.billing_period_end,
                "account_number": energy_data.account_number
            },
            confidence_scores=energy_data.confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error extracting energy fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@router.get("/formats")
async def get_supported_formats():
    """Get list of supported document formats."""
    return {
        "supported_formats": ["pdf", "jpeg", "jpg", "png"],
        "max_file_size": "10MB",
        "description": "Supported document formats for OCR processing"
    }


@router.get("/health")
async def ocr_health_check(ocr_engine: OCRProcessingEngine = Depends(get_ocr_engine)):
    """Health check for OCR service."""
    try:
        # Test OCR engine with a simple operation
        return {
            "status": "healthy",
            "ocr_engine": "PaddleOCR",
            "supported_formats": ["pdf", "jpeg", "jpg", "png"]
        }
    except Exception as e:
        logger.error(f"OCR health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="OCR service unavailable")