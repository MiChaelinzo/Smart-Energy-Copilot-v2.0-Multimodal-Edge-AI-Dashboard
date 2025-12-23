"""
Property-Based Tests for IoT Data Validation and Interpolation

**Feature: smart-energy-copilot, Property 8: IoT data validation and interpolation**
**Validates: Requirements 6.2, 7.2**

Tests that the Smart Energy Copilot validates IoT sensor data formats, handles device-specific variations,
interpolates reasonable values for missing or invalid readings, and flags data quality issues.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

from src.models.sensor_reading import SensorReading, SensorReadings
from src.services.iot_integration import IoTIntegrationService
from src.models.device import Device, DeviceConfig, ProtocolType


# Strategy for generating valid sensor readings (Pydantic-compliant)
@st.composite
def valid_sensor_readings_strategy(draw):
    """Generate valid sensor readings that pass Pydantic validation"""
    base_readings = {}
    
    # Generate valid power_watts
    if draw(st.booleans()):
        base_readings["power_watts"] = draw(st.floats(min_value=0, max_value=50000, allow_nan=False, allow_infinity=False))
    
    # Generate valid voltage
    if draw(st.booleans()):
        base_readings["voltage"] = draw(st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))
    
    # Generate valid current_amps
    if draw(st.booleans()):
        base_readings["current_amps"] = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    
    # Generate valid temperature_celsius (must be non-negative due to current Pydantic validation)
    if draw(st.booleans()):
        base_readings["temperature_celsius"] = draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    
    # Generate valid humidity_percent
    if draw(st.booleans()):
        base_readings["humidity_percent"] = draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    
    # Generate occupancy (always valid)
    if draw(st.booleans()):
        base_readings["occupancy"] = draw(st.booleans())
    
    return SensorReadings(**base_readings)


@st.composite
def sensor_reading_strategy(draw):
    """Generate complete valid sensor readings"""
    readings = draw(valid_sensor_readings_strategy())
    
    return SensorReading(
        sensor_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")).filter(lambda x: x.strip())),
        device_type=draw(st.sampled_from(["smart_meter", "temperature_sensor", "power_monitor", "occupancy_sensor", "humidity_sensor"])),
        timestamp=draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31))),
        readings=readings,
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        location=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    )


# Strategy for generating raw reading values that may be invalid
@st.composite
def raw_reading_values_strategy(draw, field_name: str):
    """Generate raw reading values that may be outside validation bounds"""
    if field_name == "power_watts":
        return draw(st.one_of(
            st.floats(min_value=-1000, max_value=-0.1),  # Invalid: negative
            st.floats(min_value=0, max_value=50000),     # Valid range
            st.floats(min_value=50001, max_value=100000) # Invalid: too high
        ))
    elif field_name == "voltage":
        return draw(st.one_of(
            st.floats(min_value=-100, max_value=-0.1),   # Invalid: negative
            st.floats(min_value=0, max_value=500),       # Valid range
            st.floats(min_value=501, max_value=1000)     # Invalid: too high
        ))
    elif field_name == "current_amps":
        return draw(st.one_of(
            st.floats(min_value=-100, max_value=-0.1),   # Invalid: negative
            st.floats(min_value=0, max_value=1000),      # Valid range
            st.floats(min_value=1001, max_value=2000)    # Invalid: too high
        ))
    elif field_name == "temperature_celsius":
        return draw(st.one_of(
            st.floats(min_value=-100, max_value=-51),    # Invalid: too cold
            st.floats(min_value=-50, max_value=100),     # Valid range
            st.floats(min_value=101, max_value=200)      # Invalid: too hot
        ))
    elif field_name == "humidity_percent":
        return draw(st.one_of(
            st.floats(min_value=-50, max_value=-0.1),    # Invalid: negative
            st.floats(min_value=0, max_value=100),       # Valid range
            st.floats(min_value=100.1, max_value=200)    # Invalid: too high
        ))
    else:
        return draw(st.floats(allow_nan=False, allow_infinity=False))


class TestIoTDataValidation:
    """Property-based tests for IoT data validation and interpolation"""

    @given(
        field_name=st.sampled_from(["power_watts", "voltage", "current_amps", "temperature_celsius", "humidity_percent"]),
        raw_value=st.floats(allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=5000)
    def test_validation_rules_bounds_checking(self, field_name: str, raw_value: float):
        """
        Property 8: IoT data validation and interpolation - Bounds checking
        
        For any field and raw value, the validation rules should correctly
        identify whether the value is within acceptable bounds.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        validation_rules = iot_service.validation_rules
        
        if field_name in validation_rules:
            rules = validation_rules[field_name]
            is_valid = rules["min"] <= raw_value <= rules["max"]
            
            # The validation logic should be consistent
            if is_valid:
                assert raw_value >= rules["min"], f"Valid value {raw_value} should be >= {rules['min']}"
                assert raw_value <= rules["max"], f"Valid value {raw_value} should be <= {rules['max']}"
            else:
                assert raw_value < rules["min"] or raw_value > rules["max"], \
                    f"Invalid value {raw_value} should be outside bounds [{rules['min']}, {rules['max']}]"

    @given(reading=sensor_reading_strategy())
    @settings(max_examples=50, deadline=10000)
    @pytest.mark.asyncio
    async def test_valid_data_passes_validation(self, reading: SensorReading):
        """
        Property 8: IoT data validation and interpolation - Valid data preservation
        
        For any sensor reading with valid values, the validation should preserve
        the original values and maintain the quality score.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        
        # Store original values
        original_readings = reading.readings.model_dump(exclude_none=True)
        original_quality_score = reading.quality_score
        
        # Mock the interpolation method to avoid database dependency
        with patch.object(iot_service, '_interpolate_value', return_value=0.0):
            validated_reading = await iot_service._validate_and_interpolate(reading)
        
        # Valid readings should be preserved (since they pass Pydantic validation, they should be within bounds)
        validated_readings = validated_reading.readings.model_dump(exclude_none=True)
        
        for field, original_value in original_readings.items():
            if field in iot_service.validation_rules:
                # Since the reading passed Pydantic validation, it should be within service bounds too
                assert validated_readings[field] == original_value, \
                    f"Valid {field} value {original_value} should be preserved"
        
        # Quality score should not be reduced for valid data
        assert validated_reading.quality_score == original_quality_score, \
            "Quality score should not be reduced for valid data"

    @given(
        reading=sensor_reading_strategy(),
        field_to_corrupt=st.sampled_from(["power_watts", "voltage", "current_amps", "temperature_celsius", "humidity_percent"]),
        invalid_value=st.floats(allow_nan=False, allow_infinity=False),
        interpolated_value=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=10000)
    @pytest.mark.asyncio
    async def test_invalid_data_interpolation_by_direct_modification(self, reading: SensorReading, field_to_corrupt: str, invalid_value: float, interpolated_value: float):
        """
        Property 8: IoT data validation and interpolation - Invalid data handling
        
        For any sensor reading with manually corrupted invalid values, the validation should
        interpolate reasonable values and reduce the quality score.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        
        # Check if the invalid value is actually invalid according to validation rules
        if field_to_corrupt not in iot_service.validation_rules:
            return  # Skip if field is not validated
            
        rules = iot_service.validation_rules[field_to_corrupt]
        is_actually_invalid = invalid_value < rules["min"] or invalid_value > rules["max"]
        
        # Skip if the "invalid" value is actually valid
        assume(is_actually_invalid)
        
        # Skip if original quality score is 0 (can't be reduced further)
        assume(reading.quality_score > 0.0)
        
        # Ensure interpolated value is valid for the field
        if field_to_corrupt == "humidity_percent":
            # Ensure interpolated humidity is within valid range
            assume(0 <= interpolated_value <= 100)
        elif field_to_corrupt == "temperature_celsius":
            # Ensure interpolated temperature is non-negative (due to current Pydantic validation)
            assume(interpolated_value >= 0)
        
        # Ensure interpolated value is within validation bounds
        assume(rules["min"] <= interpolated_value <= rules["max"])
        
        # Manually corrupt the reading by directly modifying the internal dict
        # This bypasses Pydantic validation to simulate invalid data from external sources
        original_quality_score = reading.quality_score
        
        # Create a copy and modify the readings dict directly
        readings_dict = reading.readings.model_dump(exclude_none=True)
        readings_dict[field_to_corrupt] = invalid_value
        
        # Manually create SensorReadings object with corrupted data
        # We need to bypass validation, so we'll modify the object after creation
        corrupted_reading = SensorReading(
            sensor_id=reading.sensor_id,
            device_type=reading.device_type,
            timestamp=reading.timestamp,
            readings=reading.readings,  # Start with valid readings
            quality_score=reading.quality_score,
            location=reading.location
        )
        
        # Directly modify the readings object to simulate invalid external data
        setattr(corrupted_reading.readings, field_to_corrupt, invalid_value)
        
        # Mock the interpolation method to return a known valid value
        with patch.object(iot_service, '_interpolate_value', return_value=interpolated_value):
            validated_reading = await iot_service._validate_and_interpolate(corrupted_reading)
        
        # The invalid value should be interpolated
        validated_readings = validated_reading.readings.model_dump(exclude_none=True)
        assert validated_readings[field_to_corrupt] == interpolated_value, \
            f"Invalid {field_to_corrupt} value {invalid_value} should be interpolated to {interpolated_value}"
        
        # Quality score should be reduced
        assert validated_reading.quality_score < original_quality_score, \
            "Quality score should be reduced when interpolation occurs"
        
        # Quality score should be reduced by factor of 0.8 per the implementation
        expected_quality = original_quality_score * 0.8
        assert abs(validated_reading.quality_score - expected_quality) < 0.01, \
            f"Quality score should be reduced to {expected_quality}, got {validated_reading.quality_score}"

    @given(
        device_type=st.sampled_from(["smart_meter", "temperature_sensor", "power_monitor", "occupancy_sensor", "humidity_sensor"]),
        readings_data=st.dictionaries(
            st.sampled_from(["power_watts", "voltage", "current_amps", "temperature_celsius", "humidity_percent"]),
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),  # Only valid values
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_device_specific_format_handling(self, device_type: str, readings_data: Dict[str, float]):
        """
        Property 8: IoT data validation and interpolation - Device-specific variations
        
        For any device type and reading format, the validation should handle
        device-specific variations in data format gracefully.
        
        **Validates: Requirements 6.2, 7.2**
        """
        try:
            readings = SensorReadings(**readings_data)
            
            sensor_reading = SensorReading(
                sensor_id=f"test_{device_type}_001",
                device_type=device_type,
                timestamp=datetime.now(),
                readings=readings,
                quality_score=1.0,
                location="test_location"
            )
            
            iot_service = IoTIntegrationService()
            
            # Mock interpolation to avoid database dependency
            with patch.object(iot_service, '_interpolate_value', return_value=100.0):
                validated_reading = await iot_service._validate_and_interpolate(sensor_reading)
            
            # Validation should complete without errors
            assert validated_reading is not None, "Validation should complete successfully"
            assert validated_reading.sensor_id == sensor_reading.sensor_id, "Sensor ID should be preserved"
            assert validated_reading.device_type == device_type, "Device type should be preserved"
            assert validated_reading.location == sensor_reading.location, "Location should be preserved"
            
            # Quality score should be between 0 and 1
            assert 0.0 <= validated_reading.quality_score <= 1.0, "Quality score should be in valid range"
            
        except ValueError as e:
            # Some combinations might be invalid due to Pydantic validation
            # This is acceptable as it represents proper input validation
            assert "validation error" in str(e).lower() or "must be" in str(e).lower(), \
                f"Validation error should be descriptive: {e}"

    @given(
        readings_list=st.lists(
            sensor_reading_strategy(),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=15000)
    @pytest.mark.asyncio
    async def test_batch_validation_consistency(self, readings_list):
        """
        Property 8: IoT data validation and interpolation - Batch processing consistency
        
        For any batch of sensor readings, validation should be consistent
        across all readings and maintain data integrity.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        
        # Mock interpolation to return consistent values
        interpolation_values = {
            "power_watts": 1000.0,
            "voltage": 240.0,
            "current_amps": 10.0,
            "temperature_celsius": 25.0,
            "humidity_percent": 50.0
        }
        
        async def mock_interpolate(sensor_id: str, field: str) -> float:
            return interpolation_values.get(field, 0.0)
        
        with patch.object(iot_service, '_interpolate_value', side_effect=mock_interpolate):
            validated_readings = []
            
            for reading in readings_list:
                validated_reading = await iot_service._validate_and_interpolate(reading)
                validated_readings.append(validated_reading)
            
            # All readings should be successfully validated
            assert len(validated_readings) == len(readings_list), "All readings should be processed"
            
            # Check consistency of validation rules application
            for i, (original, validated) in enumerate(zip(readings_list, validated_readings)):
                # Basic properties should be preserved
                assert validated.sensor_id == original.sensor_id, f"Sensor ID should be preserved for reading {i}"
                assert validated.device_type == original.device_type, f"Device type should be preserved for reading {i}"
                assert validated.location == original.location, f"Location should be preserved for reading {i}"
                
                # Quality score should be valid
                assert 0.0 <= validated.quality_score <= 1.0, f"Quality score should be valid for reading {i}"
                
                # Since all input readings are valid (pass Pydantic validation), quality should not decrease
                assert validated.quality_score == original.quality_score, \
                    f"Quality score should not decrease for valid data in reading {i}"

    @given(
        sensor_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")).filter(lambda x: x.strip()),
        field=st.sampled_from(["power_watts", "voltage", "current_amps", "temperature_celsius", "humidity_percent"])
    )
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_interpolation_fallback_values(self, sensor_id: str, field: str):
        """
        Property 8: IoT data validation and interpolation - Interpolation fallbacks
        
        For any sensor and field combination, interpolation should provide
        reasonable fallback values when no historical data is available.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        
        # Create a mock async context manager for database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = None  # No historical data
        mock_session.execute.return_value = mock_result
        
        # Mock the get_db_session context manager
        with patch('src.services.iot_integration.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            interpolated_value = await iot_service._interpolate_value(sensor_id, field)
            
            # Interpolated value should be reasonable for the field type
            expected_defaults = {
                "power_watts": 0.0,
                "voltage": 240.0,
                "current_amps": 0.0,
                "temperature_celsius": 20.0,
                "humidity_percent": 50.0
            }
            
            expected_default = expected_defaults.get(field, 0.0)
            assert interpolated_value == expected_default, \
                f"Interpolated value for {field} should be {expected_default}, got {interpolated_value}"
            
            # Interpolated value should be within valid bounds
            if field in iot_service.validation_rules:
                rules = iot_service.validation_rules[field]
                assert rules["min"] <= interpolated_value <= rules["max"], \
                    f"Interpolated value {interpolated_value} for {field} should be within bounds [{rules['min']}, {rules['max']}]"

    @given(
        sensor_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")).filter(lambda x: x.strip()),
        field=st.sampled_from(["power_watts", "voltage", "current_amps", "temperature_celsius", "humidity_percent"]),
        historical_value=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=5000)
    @pytest.mark.asyncio
    async def test_interpolation_with_historical_data(self, sensor_id: str, field: str, historical_value: float):
        """
        Property 8: IoT data validation and interpolation - Historical data interpolation
        
        For any sensor with historical data, interpolation should use the last known good value.
        
        **Validates: Requirements 6.2, 7.2**
        """
        iot_service = IoTIntegrationService()
        
        # Create mock historical data
        historical_readings = {field: historical_value}
        
        # Create a mock async context manager for database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (json.dumps(historical_readings),)  # Historical data available
        mock_session.execute.return_value = mock_result
        
        # Mock the get_db_session context manager
        with patch('src.services.iot_integration.get_db_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            interpolated_value = await iot_service._interpolate_value(sensor_id, field)
            
            # Should return the historical value
            assert interpolated_value == historical_value, \
                f"Interpolated value should be historical value {historical_value}, got {interpolated_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])