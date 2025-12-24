"""Property-based tests for data model validation.

**Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
**Validates: Requirements 1.3**
"""

import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from hypothesis import HealthCheck
from src.models.energy_consumption import (
    EnergyConsumption,
    DeviceConsumption, 
    SensorReading,
    BillingPeriod,
    SensorReadings
)


# Strategy generators for test data

@composite
def billing_period_strategy(draw):
    """Generate valid billing periods."""
    start_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    ))
    # End date must be after start date
    end_date = draw(st.datetimes(
        min_value=start_date + timedelta(days=1),
        max_value=start_date + timedelta(days=365)
    ))
    return BillingPeriod(start_date=start_date, end_date=end_date)


@composite
def device_consumption_strategy(draw):
    """Generate valid device consumption data."""
    return DeviceConsumption(
        device_id=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        device_type=draw(st.text(min_size=1, max_size=30).filter(lambda x: x.strip())),
        consumption_kwh=draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)),
        efficiency_rating=draw(st.one_of(st.none(), st.text(min_size=1, max_size=10))),
        usage_hours=draw(st.floats(min_value=0, max_value=24, allow_nan=False, allow_infinity=False)),
        estimated_cost=draw(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False))
    )


@composite
def sensor_readings_strategy(draw):
    """Generate valid sensor readings."""
    return SensorReadings(
        power_watts=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False))),
        voltage=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))),
        current_amps=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))),
        temperature_celsius=draw(st.one_of(st.none(), st.floats(min_value=-50, max_value=100, allow_nan=False, allow_infinity=False))),
        humidity_percent=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))),
        occupancy=draw(st.one_of(st.none(), st.booleans()))
    )


@composite
def energy_consumption_strategy(draw):
    """Generate valid energy consumption data."""
    consumption_kwh = draw(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False))
    
    # Generate device breakdown that sums to total consumption
    device_count = draw(st.integers(min_value=0, max_value=5))
    device_breakdown = None
    
    if device_count > 0:
        devices = []
        remaining_consumption = consumption_kwh
        
        for i in range(device_count - 1):
            if remaining_consumption <= 0:
                break
            device_consumption = draw(st.floats(min_value=0, max_value=remaining_consumption, allow_nan=False, allow_infinity=False))
            devices.append(DeviceConsumption(
                device_id=f"device_{i}",
                device_type=f"type_{i}",
                consumption_kwh=device_consumption,
                efficiency_rating=None,
                usage_hours=draw(st.floats(min_value=0, max_value=24, allow_nan=False, allow_infinity=False)),
                estimated_cost=draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
            ))
            remaining_consumption -= device_consumption
        
        # Last device gets remaining consumption
        if device_count > 0:
            devices.append(DeviceConsumption(
                device_id=f"device_{device_count-1}",
                device_type=f"type_{device_count-1}",
                consumption_kwh=remaining_consumption,
                efficiency_rating=None,
                usage_hours=draw(st.floats(min_value=0, max_value=24, allow_nan=False, allow_infinity=False)),
                estimated_cost=draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
            ))
        
        device_breakdown = devices
    
    return EnergyConsumption(
        id=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        timestamp=draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2024, 12, 31)
        )),
        source=draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry'])),
        consumption_kwh=consumption_kwh,
        cost_usd=draw(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False)),
        billing_period=draw(billing_period_strategy()),
        device_breakdown=device_breakdown,
        confidence_score=draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)),
        raw_data=draw(st.one_of(st.none(), st.dictionaries(st.text(), st.text())))
    )


@composite
def sensor_reading_strategy(draw):
    """Generate valid sensor reading data."""
    readings = draw(sensor_readings_strategy())
    
    # Ensure at least one reading is provided
    readings_dict = readings.model_dump()
    if not any(value is not None for value in readings_dict.values()):
        # Force at least one reading to be non-None
        readings.power_watts = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    
    return SensorReading(
        sensor_id=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        device_type=draw(st.text(min_size=1, max_size=30).filter(lambda x: x.strip())),
        timestamp=draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2024, 12, 31)
        )),
        readings=readings,
        quality_score=draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)),
        location=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    )


class TestDataModelValidation:
    """Test data model serialization round trip properties."""
    
    @given(energy_consumption_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_energy_consumption_serialization_round_trip(self, energy_consumption):
        """Property: For any valid EnergyConsumption, serializing then deserializing should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
        # **Validates: Requirements 1.3**
        
        # Serialize to dict
        serialized = energy_consumption.model_dump()
        
        # Deserialize back to object
        deserialized = EnergyConsumption(**serialized)
        
        # Should be equivalent
        assert deserialized == energy_consumption
        assert deserialized.model_dump() == energy_consumption.model_dump()
    
    @given(device_consumption_strategy())
    def test_device_consumption_serialization_round_trip(self, device_consumption):
        """Property: For any valid DeviceConsumption, serializing then deserializing should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
        # **Validates: Requirements 1.3**
        
        # Serialize to dict
        serialized = device_consumption.model_dump()
        
        # Deserialize back to object
        deserialized = DeviceConsumption(**serialized)
        
        # Should be equivalent
        assert deserialized == device_consumption
        assert deserialized.model_dump() == device_consumption.model_dump()
    
    @given(sensor_reading_strategy())
    def test_sensor_reading_serialization_round_trip(self, sensor_reading):
        """Property: For any valid SensorReading, serializing then deserializing should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
        # **Validates: Requirements 1.3**
        
        # Serialize to dict
        serialized = sensor_reading.model_dump()
        
        # Deserialize back to object
        deserialized = SensorReading(**serialized)
        
        # Should be equivalent
        assert deserialized == sensor_reading
        assert deserialized.model_dump() == sensor_reading.model_dump()
    
    @given(billing_period_strategy())
    def test_billing_period_serialization_round_trip(self, billing_period):
        """Property: For any valid BillingPeriod, serializing then deserializing should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
        # **Validates: Requirements 1.3**
        
        # Serialize to dict
        serialized = billing_period.model_dump()
        
        # Deserialize back to object
        deserialized = BillingPeriod(**serialized)
        
        # Should be equivalent
        assert deserialized == billing_period
        assert deserialized.model_dump() == billing_period.model_dump()
    
    @given(sensor_readings_strategy())
    def test_sensor_readings_serialization_round_trip(self, sensor_readings):
        """Property: For any valid SensorReadings, serializing then deserializing should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 2: Data model serialization round trip**
        # **Validates: Requirements 1.3**
        
        # Skip if no readings provided (would fail validation)
        readings_dict = sensor_readings.model_dump()
        assume(any(value is not None for value in readings_dict.values()))
        
        # Serialize to dict
        serialized = sensor_readings.model_dump()
        
        # Deserialize back to object
        deserialized = SensorReadings(**serialized)
        
        # Should be equivalent
        assert deserialized == sensor_readings
        assert deserialized.model_dump() == sensor_readings.model_dump()
