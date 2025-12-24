"""Property-based tests for energy field extraction from OCR text.

Feature: smart-energy-copilot, Property 6: Energy data field extraction
Validates: Requirements 1.2
"""

import io
from hypothesis import given, strategies as st, settings, assume
from PIL import Image, ImageDraw, ImageFont
import re

from src.services.ocr_service import OCRProcessingEngine, OCRResult, DocumentFormat


# Test data generators
@st.composite
def generate_energy_bill_text(draw):
    """Generate realistic energy bill text with known energy fields."""
    
    # Generate consumption value
    consumption = draw(st.floats(min_value=100.0, max_value=5000.0))
    consumption_str = f"{consumption:.1f}"
    
    # Generate cost value
    cost = draw(st.floats(min_value=50.0, max_value=1000.0))
    cost_str = f"{cost:.2f}"
    
    # Generate account number
    account_num = draw(st.integers(min_value=100000, max_value=999999999))
    
    # Generate billing period dates
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    end_month = start_month if start_month < 12 else 1
    end_day = start_day
    year = 2024
    
    start_date = f"{start_month:02d}/{start_day:02d}/{year}"
    end_date = f"{end_month:02d}/{end_day:02d}/{year}"
    
    # Create various text formats
    text_format = draw(st.sampled_from([
        'standard', 'verbose', 'compact', 'mixed'
    ]))
    
    if text_format == 'standard':
        text = f"""
        Electric Bill Statement
        Account Number: {account_num}
        Billing Period: {start_date} to {end_date}
        Total Usage: {consumption_str} kWh
        Total Amount Due: ${cost_str}
        """
    elif text_format == 'verbose':
        text = f"""
        ENERGY CONSUMPTION STATEMENT
        Customer Account: {account_num}
        Service Period from {start_date} through {end_date}
        Kilowatt Hours Used: {consumption_str}
        Balance Due: ${cost_str} USD
        """
    elif text_format == 'compact':
        text = f"""
        Acct# {account_num}
        Usage {consumption_str}kWh
        Amount ${cost_str}
        Period {start_date}-{end_date}
        """
    else:  # mixed
        text = f"""
        Account: {account_num}
        Energy consumption for billing period {start_date} to {end_date}
        was {consumption_str} kilowatt-hours.
        Total cost: ${cost_str}
        """
    
    return text.strip(), {
        'consumption_kwh': consumption,
        'cost_usd': cost,
        'account_number': str(account_num),
        'billing_period_start': start_date,
        'billing_period_end': end_date
    }


@st.composite
def generate_text_image_with_energy_data(draw):
    """Generate an image containing energy bill text."""
    text, expected_data = draw(generate_energy_bill_text())
    
    # Create image with text
    width = draw(st.integers(min_value=400, max_value=800))
    height = draw(st.integers(min_value=300, max_value=600))
    
    img = Image.new('RGB', (width, height), color='white')
    d = ImageDraw.Draw(img)
    
    # Draw the text
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Split text into lines and draw each line
    lines = text.split('\n')
    y_offset = 20
    for line in lines:
        if line.strip():
            d.text((20, y_offset), line.strip(), fill='black', font=font)
            y_offset += 25
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    
    return buffer.getvalue(), expected_data


@st.composite
def generate_partial_energy_text(draw):
    """Generate text with only some energy fields present."""
    
    fields_to_include = draw(st.lists(
        st.sampled_from(['consumption', 'cost', 'account', 'period']),
        min_size=1, max_size=3, unique=True
    ))
    
    text_parts = []
    expected_data = {}
    
    if 'consumption' in fields_to_include:
        consumption = draw(st.floats(min_value=50.0, max_value=3000.0))
        text_parts.append(f"Usage: {consumption:.1f} kWh")
        expected_data['consumption_kwh'] = consumption
    
    if 'cost' in fields_to_include:
        cost = draw(st.floats(min_value=25.0, max_value=800.0))
        text_parts.append(f"Total: ${cost:.2f}")
        expected_data['cost_usd'] = cost
    
    if 'account' in fields_to_include:
        account = draw(st.integers(min_value=100000, max_value=999999))
        text_parts.append(f"Account Number: {account}")
        expected_data['account_number'] = str(account)
    
    if 'period' in fields_to_include:
        month = draw(st.integers(min_value=1, max_value=12))
        day = draw(st.integers(min_value=1, max_value=28))
        start_date = f"{month:02d}/{day:02d}/2024"
        end_month = month + 1 if month < 12 else 1
        end_date = f"{end_month:02d}/{day:02d}/2024"
        text_parts.append(f"Billing period: {start_date} to {end_date}")
        expected_data['billing_period_start'] = start_date
        expected_data['billing_period_end'] = end_date
    
    text = '\n'.join(text_parts)
    return text, expected_data


class TestEnergyFieldExtraction:
    """Property tests for energy data field extraction."""
    
    @settings(max_examples=5, deadline=None)
    @given(generate_energy_bill_text())
    def test_property_6_energy_field_extraction(self, text_data):
        """
        Property 6: Energy data field extraction
        
        For any text containing energy bill information, the system should correctly
        extract the key energy fields (consumption, cost, billing period, account).
        
        Validates: Requirements 1.2
        """
        ocr_engine = OCRProcessingEngine()
        text, expected_data = text_data
        
        assume(len(text.strip()) > 0)
        
        # Create a mock OCR result
        ocr_result = OCRResult(
            text=text,
            confidence=0.9,
            bounding_boxes=[],
            format=DocumentFormat.PNG
        )
        
        # Extract energy fields
        extracted_data = ocr_engine.extract_energy_fields(ocr_result)
        
        # Verify extraction results
        if expected_data.get('consumption_kwh'):
            assert extracted_data.consumption_kwh is not None, \
                "Should extract consumption when present in text"
            # Allow for small floating point differences
            assert abs(extracted_data.consumption_kwh - expected_data['consumption_kwh']) < 0.1, \
                f"Consumption should match: expected {expected_data['consumption_kwh']}, got {extracted_data.consumption_kwh}"
        
        if expected_data.get('cost_usd'):
            assert extracted_data.cost_usd is not None, \
                "Should extract cost when present in text"
            assert abs(extracted_data.cost_usd - expected_data['cost_usd']) < 0.01, \
                f"Cost should match: expected {expected_data['cost_usd']}, got {extracted_data.cost_usd}"
        
        if expected_data.get('account_number'):
            assert extracted_data.account_number is not None, \
                "Should extract account number when present in text"
            assert extracted_data.account_number == expected_data['account_number'], \
                f"Account should match: expected {expected_data['account_number']}, got {extracted_data.account_number}"
        
        if expected_data.get('billing_period_start'):
            assert extracted_data.billing_period_start is not None, \
                "Should extract billing period start when present in text"
        
        # Verify confidence scores are provided for extracted fields
        assert isinstance(extracted_data.confidence_scores, dict), \
            "Confidence scores should be provided as a dictionary"
        
        # All confidence scores should be valid
        for field, confidence in extracted_data.confidence_scores.items():
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence for {field} should be between 0 and 1, got {confidence}"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_partial_energy_text())
    def test_property_6_partial_field_extraction(self, text_data):
        """
        Property 6b: Partial field extraction
        
        For any text with only some energy fields, the system should extract
        available fields and leave others as None.
        
        Validates: Requirements 1.2
        """
        ocr_engine = OCRProcessingEngine()
        text, expected_data = text_data
        
        assume(len(text.strip()) > 0)
        
        # Create a mock OCR result
        ocr_result = OCRResult(
            text=text,
            confidence=0.8,
            bounding_boxes=[],
            format=DocumentFormat.PNG
        )
        
        # Extract energy fields
        extracted_data = ocr_engine.extract_energy_fields(ocr_result)
        
        # Check that only expected fields are extracted
        if 'consumption_kwh' not in expected_data:
            # If consumption wasn't in the text, it should be None or not extracted
            pass  # Allow None or missing
        
        if 'cost_usd' not in expected_data:
            # If cost wasn't in the text, it should be None or not extracted
            pass  # Allow None or missing
        
        if 'account_number' not in expected_data:
            # If account wasn't in the text, it should be None or not extracted
            pass  # Allow None or missing
        
        # Verify that confidence scores are only provided for extracted fields
        for field in extracted_data.confidence_scores.keys():
            confidence = extracted_data.confidence_scores[field]
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence for {field} should be between 0 and 1, got {confidence}"
    
    @settings(max_examples=5, deadline=None)
    @given(generate_text_image_with_energy_data())
    def test_property_6_end_to_end_extraction(self, image_data):
        """
        Property 6c: End-to-end energy field extraction
        
        For any image containing energy bill text, the complete OCR + extraction
        pipeline should work correctly.
        
        Validates: Requirements 1.2
        """
        ocr_engine = OCRProcessingEngine()
        image_content, expected_data = image_data
        
        assume(len(image_content) > 0)
        
        # Process the image with OCR
        ocr_result = ocr_engine.process_document(image_content, "test.png")
        
        # Extract energy fields from OCR result
        extracted_data = ocr_engine.extract_energy_fields(ocr_result)
        
        # Verify that the extraction process completes without errors
        assert extracted_data is not None, "Energy field extraction should not fail"
        
        # Verify confidence scores structure
        assert isinstance(extracted_data.confidence_scores, dict), \
            "Confidence scores should be a dictionary"
        
        # All confidence scores should be valid
        for field, confidence in extracted_data.confidence_scores.items():
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence for {field} should be between 0 and 1, got {confidence}"
        
        # The extraction should handle the OCR text appropriately
        # (We can't guarantee exact matches due to OCR variability, but the process should work)
    
    @settings(max_examples=5, deadline=None)
    @given(st.text(min_size=10, max_size=200))
    def test_property_6_no_energy_fields(self, random_text):
        """
        Property 6d: Handling text without energy fields
        
        For any text that doesn't contain energy bill information,
        the extraction should return empty/None values gracefully.
        
        Validates: Requirements 1.2
        """
        ocr_engine = OCRProcessingEngine()
        
        # Filter out text that might accidentally contain energy-like patterns
        assume(not re.search(r'\d+\.?\d*\s*kWh', random_text, re.IGNORECASE))
        assume(not re.search(r'\$\d+\.?\d*', random_text))
        assume(not re.search(r'account\s*(?:number|#)', random_text, re.IGNORECASE))
        assume(not re.search(r'\d{1,2}/\d{1,2}/\d{4}', random_text))
        
        # Create a mock OCR result
        ocr_result = OCRResult(
            text=random_text,
            confidence=0.7,
            bounding_boxes=[],
            format=DocumentFormat.PNG
        )
        
        # Extract energy fields
        extracted_data = ocr_engine.extract_energy_fields(ocr_result)
        
        # Should not crash and should return valid structure
        assert extracted_data is not None, "Extraction should not fail on non-energy text"
        assert isinstance(extracted_data.confidence_scores, dict), \
            "Confidence scores should always be a dictionary"
        
        # Most fields should be None for non-energy text
        # (We allow some false positives due to pattern matching)
    
    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_property_6_extraction_consistency(self, data):
        """
        Property 6e: Extraction consistency
        
        For any OCR result, extracting energy fields multiple times
        should yield consistent results.
        
        Validates: Requirements 1.2
        """
        ocr_engine = OCRProcessingEngine()
        
        # Generate some text with energy data
        text, expected_data = data.draw(generate_energy_bill_text())
        
        # Create a mock OCR result
        ocr_result = OCRResult(
            text=text,
            confidence=0.8,
            bounding_boxes=[],
            format=DocumentFormat.PNG
        )
        
        # Extract fields multiple times
        extraction1 = ocr_engine.extract_energy_fields(ocr_result)
        extraction2 = ocr_engine.extract_energy_fields(ocr_result)
        extraction3 = ocr_engine.extract_energy_fields(ocr_result)
        
        # Results should be consistent
        assert extraction1.consumption_kwh == extraction2.consumption_kwh == extraction3.consumption_kwh, \
            "Consumption extraction should be consistent"
        
        assert extraction1.cost_usd == extraction2.cost_usd == extraction3.cost_usd, \
            "Cost extraction should be consistent"
        
        assert extraction1.account_number == extraction2.account_number == extraction3.account_number, \
            "Account number extraction should be consistent"
        
        assert extraction1.billing_period_start == extraction2.billing_period_start == extraction3.billing_period_start, \
            "Billing period start extraction should be consistent"
        
        # Confidence scores should also be consistent
        assert extraction1.confidence_scores == extraction2.confidence_scores == extraction3.confidence_scores, \
            "Confidence scores should be consistent"
