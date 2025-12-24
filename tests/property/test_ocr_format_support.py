"""Property-based tests for OCR multi-format document processing.

Feature: smart-energy-copilot, Property 4: Multi-format document processing
Validates: Requirements 1.5
"""

import io
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import fitz  # PyMuPDF
import numpy as np

from src.services.ocr_service import OCRProcessingEngine, DocumentFormat, OCRResult


# Test data generators
@st.composite
def generate_test_image(draw, format_type=None):
    """Generate a test image in various formats.
    
    Args:
        draw: Hypothesis draw function
        format_type: Specific format to generate (or None for random)
    """
    # Generate random image dimensions
    width = draw(st.integers(min_value=100, max_value=800))
    height = draw(st.integers(min_value=100, max_value=800))
    
    # Create a simple image with some text-like patterns
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()
    
    # Add some random dark pixels to simulate text
    num_pixels = draw(st.integers(min_value=50, max_value=500))
    for _ in range(num_pixels):
        x = draw(st.integers(min_value=0, max_value=width-1))
        y = draw(st.integers(min_value=0, max_value=height-1))
        pixels[x, y] = (0, 0, 0)
    
    # Convert to bytes in specified format
    if format_type is None:
        format_type = draw(st.sampled_from(['JPEG', 'PNG']))
    
    buffer = io.BytesIO()
    img.save(buffer, format=format_type)
    return buffer.getvalue(), format_type.lower()


@st.composite
def generate_test_pdf(draw):
    """Generate a test PDF document."""
    # Create a simple PDF with text
    pdf_buffer = io.BytesIO()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    
    # Add some text to the page
    text = draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    page.insert_text((50, 50), text)
    
    doc.save(pdf_buffer)
    doc.close()
    
    return pdf_buffer.getvalue(), 'pdf'


@st.composite
def generate_document(draw):
    """Generate a document in any supported format."""
    doc_type = draw(st.sampled_from(['image', 'pdf']))
    
    if doc_type == 'pdf':
        return draw(generate_test_pdf())
    else:
        return draw(generate_test_image())


class TestOCRFormatSupport:
    """Property tests for multi-format document processing."""
    
    def setup_method(self):
        """Set up test fixtures with mocked PaddleOCR."""
        # Mock PaddleOCR since it's not installed
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)],
                    [[[0, 25], [150, 25], [150, 45], [0, 45]], ('Energy usage: 123 kWh', 0.88)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            self.ocr_engine = OCRProcessingEngine()
    
    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_property_4_multi_format_processing(self, data):
        """
        Property 4: Multi-format document processing
        
        For any document in a supported format (PDF, JPEG, PNG),
        the OCR engine should successfully process it and return a valid OCR result.
        
        Validates: Requirements 1.5
        """
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)],
                    [[[0, 25], [150, 25], [150, 45], [0, 45]], ('Energy usage: 123 kWh', 0.88)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            ocr_engine = OCRProcessingEngine()
            
            # Generate a document in a random supported format
            doc_content, doc_format = data.draw(generate_document())
            
            # Assume we have valid content
            assume(len(doc_content) > 0)
            
            # Process the document
            result = ocr_engine.process_document(doc_content, f"test.{doc_format}")
            
            # Verify the result is valid
            assert result is not None, "OCR result should not be None"
            assert result.format.value in ['pdf', 'jpeg', 'jpg', 'png'], \
                f"Format should be supported, got {result.format.value}"
            assert result.confidence >= 0.0 and result.confidence <= 1.0, \
                f"Confidence should be between 0 and 1, got {result.confidence}"
            assert isinstance(result.text, str), "Text should be a string"
            assert isinstance(result.bounding_boxes, list), "Bounding boxes should be a list"
            assert result.page_count >= 1, "Page count should be at least 1"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_test_image(format_type='JPEG'))
    def test_jpeg_format_processing(self, image_data):
        """Test JPEG format processing specifically."""
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            ocr_engine = OCRProcessingEngine()
            
            content, format_type = image_data
            
            result = ocr_engine.process_document(content, "test.jpg")
            
            assert result.format in [DocumentFormat.JPEG, DocumentFormat.JPG], \
                f"Should detect JPEG format, got {result.format}"
            assert result.page_count == 1, "Image should have 1 page"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_test_image(format_type='PNG'))
    def test_png_format_processing(self, image_data):
        """Test PNG format processing specifically."""
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            ocr_engine = OCRProcessingEngine()
            
            content, format_type = image_data
            
            result = ocr_engine.process_document(content, "test.png")
            
            assert result.format == DocumentFormat.PNG, \
                f"Should detect PNG format, got {result.format}"
            assert result.page_count == 1, "Image should have 1 page"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_test_pdf())
    def test_pdf_format_processing(self, pdf_data):
        """Test PDF format processing specifically."""
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            ocr_engine = OCRProcessingEngine()
            
            content, format_type = pdf_data
            
            result = ocr_engine.process_document(content, "test.pdf")
            
            assert result.format == DocumentFormat.PDF, \
                f"Should detect PDF format, got {result.format}"
            assert result.page_count >= 1, "PDF should have at least 1 page"
    
    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_format_detection_consistency(self, data):
        """
        Property: Format detection should be consistent.
        
        For any document, detecting the format multiple times should yield the same result.
        """
        with patch('src.services.ocr_service.PaddleOCR') as mock_paddle:
            mock_instance = Mock()
            mock_instance.predict.return_value = [
                [
                    [[[0, 0], [100, 0], [100, 20], [0, 20]], ('Sample text', 0.95)]
                ]
            ]
            mock_paddle.return_value = mock_instance
            ocr_engine = OCRProcessingEngine()
            
            doc_content, doc_format = data.draw(generate_document())
            assume(len(doc_content) > 0)
            
            # Detect format multiple times
            format1 = ocr_engine.detect_format(doc_content, f"test.{doc_format}")
            format2 = ocr_engine.detect_format(doc_content, f"test.{doc_format}")
            format3 = ocr_engine.detect_format(doc_content, f"test.{doc_format}")
            
            # All detections should be the same
            assert format1 == format2 == format3, \
                "Format detection should be consistent across multiple calls"
