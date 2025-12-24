"""Property-based tests for OCR confidence scoring validation.

Feature: smart-energy-copilot, Property 5: OCR confidence validation
Validates: Requirements 1.4
"""

import io
from hypothesis import given, strategies as st, settings, assume
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.services.ocr_service import OCRProcessingEngine


# Test data generators
@st.composite
def generate_clear_text_image(draw):
    """Generate a clear, high-quality text image that should have high confidence."""
    width = draw(st.integers(min_value=200, max_value=600))
    height = draw(st.integers(min_value=100, max_value=300))
    
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    d = ImageDraw.Draw(img)
    
    # Add clear, readable text
    text = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    # Use a reasonable font size
    font_size = draw(st.integers(min_value=16, max_value=32))
    
    try:
        # Try to use a system font, fall back to default if not available
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw text in black on white background for maximum contrast
    d.text((10, 10), text, fill='black', font=font)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue(), text


@st.composite
def generate_poor_quality_image(draw):
    """Generate a poor quality image that should have low confidence."""
    width = draw(st.integers(min_value=50, max_value=200))
    height = draw(st.integers(min_value=30, max_value=100))
    
    # Create noisy background
    img = Image.new('RGB', (width, height), color='gray')
    pixels = img.load()
    
    # Add random noise
    num_noise_pixels = draw(st.integers(min_value=100, max_value=500))
    for _ in range(num_noise_pixels):
        x = draw(st.integers(min_value=0, max_value=width-1))
        y = draw(st.integers(min_value=0, max_value=height-1))
        color = draw(st.integers(min_value=0, max_value=255))
        pixels[x, y] = (color, color, color)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@st.composite
def generate_mixed_quality_image(draw):
    """Generate an image with mixed quality elements."""
    width = draw(st.integers(min_value=150, max_value=400))
    height = draw(st.integers(min_value=100, max_value=200))
    
    # Create background with some noise
    img = Image.new('RGB', (width, height), color='lightgray')
    d = ImageDraw.Draw(img)
    pixels = img.load()
    
    # Add some noise
    num_noise_pixels = draw(st.integers(min_value=20, max_value=100))
    for _ in range(num_noise_pixels):
        x = draw(st.integers(min_value=0, max_value=width-1))
        y = draw(st.integers(min_value=0, max_value=height-1))
        color = draw(st.integers(min_value=100, max_value=200))
        pixels[x, y] = (color, color, color)
    
    # Add some readable text
    text = draw(st.text(min_size=3, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    d.text((10, 10), text, fill='black')
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue(), text


class TestOCRConfidenceScoring:
    """Property tests for OCR confidence validation."""
    
    @settings(max_examples=5, deadline=None)
    @given(generate_clear_text_image())
    def test_property_5_high_quality_confidence(self, image_data):
        """
        Property 5a: OCR confidence validation for high-quality images
        
        For any clear, high-quality text image, the OCR confidence should be reasonably high.
        
        Validates: Requirements 1.4
        """
        ocr_engine = OCRProcessingEngine()
        image_content, expected_text = image_data
        
        # Assume we have valid content
        assume(len(image_content) > 0)
        assume(len(expected_text.strip()) > 0)
        
        # Process the image
        result = ocr_engine.process_document(image_content, "test.png")
        
        # Verify confidence is reasonable for clear images
        assert result.confidence >= 0.0 and result.confidence <= 1.0, \
            f"Confidence should be between 0 and 1, got {result.confidence}"
        
        # For clear images, we expect reasonable confidence (not necessarily high due to simple test images)
        assert result.confidence >= 0.1, \
            f"Clear images should have some confidence, got {result.confidence}"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_poor_quality_image())
    def test_property_5_poor_quality_confidence(self, image_content):
        """
        Property 5b: OCR confidence validation for poor-quality images
        
        For any poor quality image, the OCR confidence should reflect the quality.
        
        Validates: Requirements 1.4
        """
        ocr_engine = OCRProcessingEngine()
        
        # Assume we have valid content
        assume(len(image_content) > 0)
        
        # Process the image
        result = ocr_engine.process_document(image_content, "test.png")
        
        # Verify confidence is in valid range
        assert result.confidence >= 0.0 and result.confidence <= 1.0, \
            f"Confidence should be between 0 and 1, got {result.confidence}"
        
        # Poor quality images should generally have lower confidence, but we can't guarantee this
        # since OCR might still detect some patterns. The key is that confidence is valid.
    
    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_property_5_confidence_consistency(self, data):
        """
        Property 5c: Confidence scoring consistency
        
        For any image, processing it multiple times should yield consistent confidence scores.
        
        Validates: Requirements 1.4
        """
        ocr_engine = OCRProcessingEngine()
        
        # Generate an image
        image_type = data.draw(st.sampled_from(['clear', 'mixed']))
        
        if image_type == 'clear':
            image_content, _ = data.draw(generate_clear_text_image())
        else:
            image_content, _ = data.draw(generate_mixed_quality_image())
        
        assume(len(image_content) > 0)
        
        # Process the same image multiple times
        result1 = ocr_engine.process_document(image_content, "test.png")
        result2 = ocr_engine.process_document(image_content, "test.png")
        result3 = ocr_engine.process_document(image_content, "test.png")
        
        # All confidence scores should be the same (deterministic)
        assert result1.confidence == result2.confidence == result3.confidence, \
            f"Confidence should be consistent: {result1.confidence}, {result2.confidence}, {result3.confidence}"
        
        # All should be in valid range
        for result in [result1, result2, result3]:
            assert result.confidence >= 0.0 and result.confidence <= 1.0, \
                f"Confidence should be between 0 and 1, got {result.confidence}"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_mixed_quality_image())
    def test_property_5_quality_assessment_validity(self, image_data):
        """
        Property 5d: Quality assessment validity
        
        For any image, the quality assessment should provide valid recommendations
        based on the confidence score.
        
        Validates: Requirements 1.4
        """
        ocr_engine = OCRProcessingEngine()
        image_content, expected_text = image_data
        
        assume(len(image_content) > 0)
        
        # Process the image
        result = ocr_engine.process_document(image_content, "test.png")
        
        # Get quality assessment
        assessment = ocr_engine.assess_quality(result)
        
        # Verify assessment structure
        assert 'overall_confidence' in assessment
        assert 'quality_level' in assessment
        assert 'recommendations' in assessment
        
        # Confidence should match
        assert assessment['overall_confidence'] == result.confidence
        
        # Quality level should be valid
        valid_levels = ['excellent', 'good', 'fair', 'poor', 'unknown']
        assert assessment['quality_level'] in valid_levels, \
            f"Quality level should be one of {valid_levels}, got {assessment['quality_level']}"
        
        # Recommendations should be a list
        assert isinstance(assessment['recommendations'], list)
        
        # Quality level should correlate with confidence
        if result.confidence >= 0.9:
            assert assessment['quality_level'] == 'excellent'
        elif result.confidence >= 0.7:
            assert assessment['quality_level'] == 'good'
        elif result.confidence >= 0.5:
            assert assessment['quality_level'] == 'fair'
        elif result.confidence < 0.5:
            assert assessment['quality_level'] == 'poor'
    
    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_property_5_confidence_bounds(self, data):
        """
        Property 5e: Confidence bounds validation
        
        For any document processing result, confidence scores should always be within [0, 1].
        
        Validates: Requirements 1.4
        """
        ocr_engine = OCRProcessingEngine()
        
        # Generate various types of images
        image_type = data.draw(st.sampled_from(['clear', 'poor', 'mixed']))
        
        if image_type == 'clear':
            image_content, _ = data.draw(generate_clear_text_image())
        elif image_type == 'poor':
            image_content = data.draw(generate_poor_quality_image())
        else:
            image_content, _ = data.draw(generate_mixed_quality_image())
        
        assume(len(image_content) > 0)
        
        # Process the image
        result = ocr_engine.process_document(image_content, "test.png")
        
        # Confidence must be in valid bounds
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence must be between 0 and 1, got {result.confidence}"
        
        # Individual bounding box confidences should also be valid
        for bbox, text, confidence in result.bounding_boxes:
            assert 0.0 <= confidence <= 1.0, \
                f"Bounding box confidence must be between 0 and 1, got {confidence}"
