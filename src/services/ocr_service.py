"""OCR processing service for document text extraction and energy field identification."""

import io
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF processing

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

from src.config.logging import get_logger
from src.services.error_handling import (
    with_error_handling, OCRProcessingError, ValidationError,
    ErrorContext, ErrorSeverity, error_handler
)

logger = get_logger(__name__)


class DocumentFormat(Enum):
    """Supported document formats."""
    PDF = "pdf"
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"


@dataclass
class OCRResult:
    """OCR processing result with confidence scoring."""
    text: str
    confidence: float
    bounding_boxes: List[Tuple[List[List[int]], str, float]]
    format: DocumentFormat
    page_count: int = 1


@dataclass
class EnergyFieldData:
    """Extracted energy-related data fields."""
    consumption_kwh: Optional[float] = None
    cost_usd: Optional[float] = None
    billing_period_start: Optional[str] = None
    billing_period_end: Optional[str] = None
    account_number: Optional[str] = None
    service_address: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}


class OCRProcessingEngine:
    """PaddleOCR-based document processing engine for energy documents."""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        """Initialize OCR engine.
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language for OCR recognition
        """
        if PADDLEOCR_AVAILABLE and PaddleOCR is not None:
            self.ocr = PaddleOCR(
                use_textline_orientation=use_angle_cls,
                lang=lang
            )
        else:
            # For testing without PaddleOCR installed
            self.ocr = None
            logger.warning("PaddleOCR not available, OCR functionality will be limited")
        
        self.supported_formats = {fmt.value for fmt in DocumentFormat}
        
        # Energy field patterns for extraction
        self.energy_patterns = {
            'consumption_kwh': [
                r'usage:?\s*(\d+(?:\.\d+)?)\s*kWh',
                r'kilowatt\s*hours?\s*used:?\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*kWh',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*kilowatt.?hours?',
                r'total\s*usage:?\s*(\d+(?:\.\d+)?)',
            ],
            'cost_usd': [
                r'amount:?\s*\$(\d+(?:\.\d{2})?)',
                r'balance\s*due:?\s*\$(\d+(?:\.\d{2})?)',
                r'total\s*amount:?\s*\$?(\d+(?:\.\d{2})?)',
                r'\$(\d+(?:\.\d{2})?)',
                r'(\d+(?:\.\d{2})?)\s*USD',
            ],
            'billing_period': [
                r'period:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*[-–]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'billing\s*period:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|-)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'service\s*(?:period\s*)?from:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|through)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'account_number': [
                r'account:?\s*(\d+)',
                r'account\s*(?:number|#):?\s*(\d+)',
                r'acct\s*(?:no|#):?\s*(\d+)',
                r'customer\s*account:?\s*(\d+)',
            ]
        }
    
    @with_error_handling("ocr_service", "detect_format")
    def detect_format(self, file_content: bytes, filename: str = "") -> DocumentFormat:
        """Detect document format from content and filename.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename for format hint
            
        Returns:
            Detected document format
            
        Raises:
            ValueError: If format is not supported
        """
        # Check filename extension first
        if filename:
            ext = Path(filename).suffix.lower().lstrip('.')
            if ext in self.supported_formats:
                return DocumentFormat(ext)
        
        # Check file signature/magic bytes
        if file_content.startswith(b'%PDF'):
            return DocumentFormat.PDF
        elif file_content.startswith(b'\xff\xd8\xff'):
            return DocumentFormat.JPEG
        elif file_content.startswith(b'\x89PNG'):
            return DocumentFormat.PNG
        
        # Try to open as image
        try:
            Image.open(io.BytesIO(file_content))
            return DocumentFormat.JPEG  # Default to JPEG for unknown image formats
        except Exception as e:
            logger.error(f"Failed to detect image format: {e}")
            
        raise OCRProcessingError("Unsupported document format. Please use PDF, JPEG, or PNG files.")
    
    @with_error_handling("ocr_service", "process_document")
    def process_document(self, file_content: bytes, filename: str = "") -> OCRResult:
        """Process document and extract text with OCR.
        
        Args:
            file_content: Raw document bytes
            filename: Original filename
            
        Returns:
            OCR processing result with confidence scores
        """
        try:
            doc_format = self.detect_format(file_content, filename)
            logger.info(f"Processing document format: {doc_format.value}")
            
            if doc_format == DocumentFormat.PDF:
                return self._process_pdf(file_content, doc_format)
            else:
                return self._process_image(file_content, doc_format)
                
        except OCRProcessingError:
            # Re-raise OCR-specific errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document: {str(e)}")
            raise OCRProcessingError(f"Failed to process document: {str(e)}")
    
    def _process_pdf(self, file_content: bytes, doc_format: DocumentFormat) -> OCRResult:
        """Process PDF document."""
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        all_text = []
        all_boxes = []
        total_confidence = 0.0
        confidence_count = 0
        
        try:
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Process with OCR
                if self.ocr is None:
                    # Fallback for testing
                    result = [
                        [
                            [[[0, 0], [100, 0], [100, 20], [0, 20]], ("Mock PDF text", 0.95)]
                        ]
                    ]
                else:
                    try:
                        # Convert PNG bytes to PIL Image, then to numpy array
                        image = Image.open(io.BytesIO(img_data))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        img_array = np.array(image)
                        result = self.ocr.predict(img_array)
                    except Exception as e:
                        logger.error(f"Error processing PDF page {page_num} with OCR: {e}")
                        result = []
                
                if result and result[0]:
                    page_text = []
                    for line in result[0]:
                        if len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            
                            # Handle different PaddleOCR result formats
                            if isinstance(text_info, tuple) and len(text_info) == 2:
                                text, confidence = text_info
                            elif isinstance(text_info, str):
                                text = text_info
                                confidence = 0.5  # Default confidence when not provided
                            else:
                                # Skip malformed results
                                continue
                                
                            page_text.append(text)
                            all_boxes.append((bbox, text, confidence))
                            total_confidence += confidence
                            confidence_count += 1
                    
                    all_text.extend(page_text)
            
            overall_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
            
            return OCRResult(
                text=" ".join(all_text),
                confidence=overall_confidence,
                bounding_boxes=all_boxes,
                format=doc_format,
                page_count=pdf_doc.page_count
            )
            
        finally:
            pdf_doc.close()
    
    def _process_image(self, file_content: bytes, doc_format: DocumentFormat) -> OCRResult:
        """Process image document."""
        if self.ocr is None:
            # Fallback for testing without PaddleOCR
            return OCRResult(
                text="Mock OCR text for testing",
                confidence=0.95,
                bounding_boxes=[([[[0, 0], [100, 0], [100, 20], [0, 20]]], "Mock text", 0.95)],
                format=doc_format
            )
        
        try:
            # Convert bytes to PIL Image, then to numpy array
            image = Image.open(io.BytesIO(file_content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            img_array = np.array(image)
            
            result = self.ocr.predict(img_array)
        except Exception as e:
            logger.error(f"Error processing image with OCR: {e}")
            # Return empty result on error
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                format=doc_format
            )
        
        if not result or not result[0]:
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                format=doc_format
            )
        
        text_parts = []
        all_boxes = []
        total_confidence = 0.0
        confidence_count = 0
        
        for line in result[0]:
            if len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                
                # Handle different PaddleOCR result formats
                if isinstance(text_info, tuple) and len(text_info) == 2:
                    text, confidence = text_info
                elif isinstance(text_info, str):
                    text = text_info
                    confidence = 0.5  # Default confidence when not provided
                else:
                    # Skip malformed results
                    continue
                    
                text_parts.append(text)
                all_boxes.append((bbox, text, confidence))
                total_confidence += confidence
                confidence_count += 1
        
        overall_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        return OCRResult(
            text=" ".join(text_parts),
            confidence=overall_confidence,
            bounding_boxes=all_boxes,
            format=doc_format
        )
    
    def extract_energy_fields(self, ocr_result: OCRResult) -> EnergyFieldData:
        """Extract structured energy data from OCR text.
        
        Args:
            ocr_result: OCR processing result
            
        Returns:
            Structured energy field data with confidence scores
        """
        text = ocr_result.text.lower()
        energy_data = EnergyFieldData()
        
        # Extract consumption in kWh
        for pattern in self.energy_patterns['consumption_kwh']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    energy_data.consumption_kwh = value
                    energy_data.confidence_scores['consumption_kwh'] = self._calculate_field_confidence(
                        match.group(0), ocr_result.bounding_boxes
                    )
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract cost in USD
        for pattern in self.energy_patterns['cost_usd']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    energy_data.cost_usd = value
                    energy_data.confidence_scores['cost_usd'] = self._calculate_field_confidence(
                        match.group(0), ocr_result.bounding_boxes
                    )
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract billing period
        for pattern in self.energy_patterns['billing_period']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    energy_data.billing_period_start = match.group(1)
                    if len(match.groups()) > 1:
                        energy_data.billing_period_end = match.group(2)
                    energy_data.confidence_scores['billing_period'] = self._calculate_field_confidence(
                        match.group(0), ocr_result.bounding_boxes
                    )
                    break
                except IndexError:
                    continue
        
        # Extract account number
        for pattern in self.energy_patterns['account_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    energy_data.account_number = match.group(1)
                    energy_data.confidence_scores['account_number'] = self._calculate_field_confidence(
                        match.group(0), ocr_result.bounding_boxes
                    )
                    break
                except IndexError:
                    continue
        
        return energy_data
    
    def _calculate_field_confidence(self, matched_text: str, bounding_boxes: List[Tuple]) -> float:
        """Calculate confidence score for extracted field based on OCR confidence.
        
        Args:
            matched_text: The text that was matched
            bounding_boxes: OCR bounding boxes with confidence scores
            
        Returns:
            Confidence score for the field (0.0 to 1.0)
        """
        # Find bounding boxes that contain parts of the matched text
        relevant_confidences = []
        
        for bbox, text, confidence in bounding_boxes:
            if any(word in text.lower() for word in matched_text.lower().split()):
                relevant_confidences.append(confidence)
        
        if relevant_confidences:
            return sum(relevant_confidences) / len(relevant_confidences)
        else:
            return 0.5  # Default confidence if no specific match found
    
    def assess_quality(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Assess document quality and provide recommendations.
        
        Args:
            ocr_result: OCR processing result
            
        Returns:
            Quality assessment with recommendations
        """
        assessment = {
            'overall_confidence': ocr_result.confidence,
            'text_length': len(ocr_result.text),
            'word_count': len(ocr_result.text.split()),
            'quality_level': 'unknown',
            'recommendations': []
        }
        
        # Determine quality level
        if ocr_result.confidence >= 0.9:
            assessment['quality_level'] = 'excellent'
        elif ocr_result.confidence >= 0.7:
            assessment['quality_level'] = 'good'
        elif ocr_result.confidence >= 0.5:
            assessment['quality_level'] = 'fair'
            assessment['recommendations'].append('Consider rescanning with higher resolution')
        else:
            assessment['quality_level'] = 'poor'
            assessment['recommendations'].extend([
                'Document quality is poor - consider rescanning',
                'Ensure document is well-lit and in focus',
                'Try straightening the document if skewed'
            ])
        
        # Check for insufficient text
        if assessment['word_count'] < 10:
            assessment['recommendations'].append('Very little text detected - verify document content')
        
        return assessment