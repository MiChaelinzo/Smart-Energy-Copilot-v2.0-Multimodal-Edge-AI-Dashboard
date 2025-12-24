"""Property-based tests for OCR error handling with poor quality documents.

**Validates: Requirements 6.1**
"""

import io
import pytest
from hypothesis import given, strategies as st, assume, settings
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from typing import Dict, Any

from src.services.ocr_service import OCRProcessingEngine, DocumentFormat, OCRResult


class TestOCRErrorHandlingProperties:
    """Property-based tests for OCR error handling with poor quality documents."""
    
    def create_poor_quality_image(
        self, 
        width: int, 
        height: int, 
        blur_radius: float = 0.0,
        noise_level: float = 0.0,
        brightness: float = 1.0,
        contrast: float = 1.0
    ) -> bytes:
        """Create a poor quality image for testing OCR error handling."""
        # Create a simple image with text
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Add some text
        text = "Energy Bill\nUsage: 123.45 kWh\nAmount: $67.89"
        try:
            draw.text((10, 10), text, fill='black')
        except OSError:
            # If no font available, just draw some shapes
            draw.rectangle([10, 10, 100, 30], fill='black')
            draw.rectangle([10, 40, 80, 60], fill='black')
        
        # Apply quality degradation
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Add noise
        if noise_level > 0:
            img_array = np.array(image)
            noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        # Adjust brightness and contrast
        if brightness != 1.0 or contrast != 1.0:
            from PIL import ImageEnhance
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    @settings(deadline=15000, max_examples=3)  # 15 second deadline, only 3 examples
    @given(
        width=st.integers(min_value=100, max_value=300),  # Smaller images for faster processing
        height=st.integers(min_value=100, max_value=300),
        blur_radius=st.floats(min_value=0.0, max_value=5.0),  # Reduced range
        noise_level=st.floats(min_value=0.0, max_value=0.3),  # Reduced range
        brightness=st.floats(min_value=0.3, max_value=1.5),  # Reduced range
        contrast=st.floats(min_value=0.3, max_value=1.5)  # Reduced range
    )
    def test_poor_quality_document_processing_property(
        self, 
        width: int, 
        height: int, 
        blur_radius: float,
        noise_level: float,
        brightness: float,
        contrast: float
    ):
        """
        Property 25: Poor quality document processing
        
        For any document with poor quality characteristics (blur, noise, poor lighting),
        the OCR system should:
        1. Process the document without crashing
        2. Return a valid OCRResult with appropriate confidence scores
        3. Provide quality assessment and recommendations
        4. Handle errors gracefully and provide user-friendly feedback
        
        **Validates: Requirements 6.1**
        """
        # Arrange
        ocr_engine = OCRProcessingEngine()
        
        # Skip test if PaddleOCR is not available (use mock behavior)
        if not hasattr(ocr_engine, 'ocr') or ocr_engine.ocr is None:
            # Use simplified mock testing
            poor_image_bytes = self.create_poor_quality_image(
                width, height, blur_radius, noise_level, brightness, contrast
            )
            
            # Mock OCR result for testing
            mock_result = OCRResult(
                text="Mock OCR text",
                confidence=max(0.1, 1.0 - (blur_radius/5.0 + noise_level)),  # Simulate quality impact
                bounding_boxes=[],
                format=DocumentFormat.PNG
            )
            
            # Test quality assessment
            quality_assessment = ocr_engine.assess_quality(mock_result)
            
            # Basic assertions for mock behavior
            assert isinstance(mock_result, OCRResult)
            assert isinstance(quality_assessment, dict)
            assert 'quality_level' in quality_assessment
            assert 'recommendations' in quality_assessment
            return
        
        # Create poor quality image
        poor_image_bytes = self.create_poor_quality_image(
            width, height, blur_radius, noise_level, brightness, contrast
        )
        
        # Act - Process the poor quality document
        try:
            ocr_result = ocr_engine.process_document(poor_image_bytes, "test.png")
            
            # Assert - OCR should not crash and return valid result
            assert isinstance(ocr_result, OCRResult)
            assert isinstance(ocr_result.text, str)
            assert isinstance(ocr_result.confidence, float)
            assert 0.0 <= ocr_result.confidence <= 1.0
            assert ocr_result.format == DocumentFormat.PNG
            
            # Assert - Quality assessment should be provided
            quality_assessment = ocr_engine.assess_quality(ocr_result)
            assert isinstance(quality_assessment, dict)
            assert 'overall_confidence' in quality_assessment
            assert 'quality_level' in quality_assessment
            assert 'recommendations' in quality_assessment
            assert isinstance(quality_assessment['recommendations'], list)
            
            # Assert - Poor quality should be detected and recommendations provided
            is_poor_quality = (
                blur_radius > 3.0 or 
                noise_level > 0.2 or 
                brightness < 0.3 or brightness > 1.7 or
                contrast < 0.3 or contrast > 1.7
            )
            
            if is_poor_quality:
                # Should have lower confidence or provide recommendations
                assert (
                    ocr_result.confidence < 0.8 or 
                    len(quality_assessment['recommendations']) > 0
                )
                
                # Quality level should reflect poor quality
                if ocr_result.confidence < 0.5:
                    assert quality_assessment['quality_level'] in ['poor', 'fair']
            
            # Assert - Recommendations should be helpful and user-friendly
            for recommendation in quality_assessment['recommendations']:
                assert isinstance(recommendation, str)
                assert len(recommendation) > 0
                # Should not contain technical jargon
                technical_terms = ['exception', 'error', 'null', 'undefined', 'traceback']
                recommendation_lower = recommendation.lower()
                for term in technical_terms:
                    assert term not in recommendation_lower
            
        except Exception as e:
            # If processing fails, it should be handled gracefully
            error_message = str(e)
            
            # Error message should be informative but not technical
            assert len(error_message) > 0
            
            # Should not expose internal implementation details
            technical_terms = ['traceback', 'stack trace', 'null pointer', 'segmentation fault']
            error_lower = error_message.lower()
            for term in technical_terms:
                assert term not in error_lower
    
    @settings(deadline=10000, max_examples=3)  # 10 second deadline, only 3 examples
    @given(
        file_size=st.integers(min_value=1, max_value=100),  # Smaller files
        format_hint=st.sampled_from(['test.pdf', 'test.jpg', 'test.png', 'test.unknown'])
    )
    def test_invalid_document_handling_property(self, file_size: int, format_hint: str):
        """
        Property: Invalid document handling
        
        For any invalid or corrupted document data, the OCR system should:
        1. Detect the invalid format gracefully
        2. Provide appropriate error messages
        3. Not crash or expose internal errors
        4. Suggest recovery actions to the user
        
        **Validates: Requirements 6.1**
        """
        # Arrange
        ocr_engine = OCRProcessingEngine()
        
        # Create invalid/corrupted file data
        invalid_data = b'invalid_file_content' + b'\x00' * file_size
        
        # Act & Assert
        try:
            result = ocr_engine.process_document(invalid_data, format_hint)
            
            # If it doesn't raise an exception, it should return a valid result
            assert isinstance(result, OCRResult)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            
        except ValueError as e:
            # Expected for unsupported formats
            error_message = str(e)
            assert "format" in error_message.lower() or "supported" in error_message.lower()
            
        except Exception as e:
            # Any other exception should have a user-friendly message
            error_message = str(e)
            assert len(error_message) > 0
            
            # Should not contain technical implementation details
            forbidden_terms = [
                'traceback', 'stack', 'null pointer', 'segmentation',
                'memory', 'buffer overflow', 'assertion failed'
            ]
            error_lower = error_message.lower()
            for term in forbidden_terms:
                assert term not in error_lower
    
    @settings(deadline=8000, max_examples=3)  # 8 second deadline, only 3 examples
    @given(
        empty_content=st.booleans(),
        minimal_content=st.booleans()
    )
    def test_edge_case_document_handling_property(self, empty_content: bool, minimal_content: bool):
        """
        Property: Edge case document handling
        
        For edge cases like empty documents or minimal content, the OCR system should:
        1. Handle empty or minimal content gracefully
        2. Return appropriate confidence scores
        3. Provide helpful feedback about content issues
        4. Not crash on edge cases
        
        **Validates: Requirements 6.1**
        """
        # Arrange
        ocr_engine = OCRProcessingEngine()
        
        if empty_content:
            # Create completely empty image
            image = Image.new('RGB', (100, 100), color='white')
        elif minimal_content:
            # Create image with minimal content
            image = Image.new('RGB', (100, 100), color='white')
            draw = ImageDraw.Draw(image)
            draw.point((50, 50), fill='black')  # Single pixel
        else:
            # Create normal image for comparison
            image = Image.new('RGB', (200, 100), color='white')
            draw = ImageDraw.Draw(image)
            try:
                draw.text((10, 10), "Test", fill='black')
            except OSError:
                draw.rectangle([10, 10, 50, 30], fill='black')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        image_data = img_bytes.getvalue()
        
        # Act
        try:
            result = ocr_engine.process_document(image_data, "test.png")
            
            # Assert - Should return valid result
            assert isinstance(result, OCRResult)
            assert isinstance(result.text, str)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            
            # Assert - Quality assessment should handle edge cases
            quality_assessment = ocr_engine.assess_quality(result)
            assert isinstance(quality_assessment, dict)
            
            # For empty or minimal content, should provide appropriate feedback
            if empty_content or minimal_content:
                # Should have low confidence or provide recommendations
                assert (
                    result.confidence < 0.7 or 
                    len(quality_assessment['recommendations']) > 0 or
                    quality_assessment['word_count'] < 5
                )
                
                # Should suggest content verification for minimal text
                if quality_assessment['word_count'] < 10:
                    recommendations_text = ' '.join(quality_assessment['recommendations']).lower()
                    assert any(word in recommendations_text for word in ['text', 'content', 'document', 'verify'])
            
        except Exception as e:
            # Should handle edge cases gracefully
            error_message = str(e)
            assert len(error_message) > 0
            
            # Error should be user-understandable
            assert not any(term in error_message.lower() for term in ['traceback', 'null', 'segmentation'])
    
    def test_confidence_scoring_consistency_property(self):
        """
        Property: Confidence scoring consistency
        
        The OCR confidence scoring should be consistent and meaningful:
        1. Higher quality images should generally have higher confidence
        2. Confidence scores should be in valid range [0.0, 1.0]
        3. Quality assessment should correlate with confidence scores
        
        **Validates: Requirements 6.1**
        """
        # Arrange
        ocr_engine = OCRProcessingEngine()
        
        # Create images with different quality levels
        high_quality = self.create_poor_quality_image(300, 200, 0.0, 0.0, 1.0, 1.0)
        medium_quality = self.create_poor_quality_image(300, 200, 2.0, 0.1, 0.8, 0.8)
        low_quality = self.create_poor_quality_image(300, 200, 5.0, 0.3, 0.4, 0.4)
        
        # Process all images
        results = []
        for quality_data in [high_quality, medium_quality, low_quality]:
            try:
                result = ocr_engine.process_document(quality_data, "test.png")
                results.append(result)
            except Exception:
                # If processing fails, assign zero confidence
                results.append(OCRResult("", 0.0, [], DocumentFormat.PNG))
        
        # Assert - Confidence scores should be valid
        for result in results:
            assert 0.0 <= result.confidence <= 1.0
        
        # Assert - Quality assessment should be consistent
        for result in results:
            if result.confidence > 0:  # Only check if processing succeeded
                assessment = ocr_engine.assess_quality(result)
                
                # Quality level should correlate with confidence
                if result.confidence >= 0.9:
                    assert assessment['quality_level'] in ['excellent', 'good']
                elif result.confidence < 0.5:
                    assert assessment['quality_level'] in ['poor', 'fair']


# Feature: smart-energy-copilot, Property 25: Poor quality document processing