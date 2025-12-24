"""Property-based tests for data storage consistency.

**Feature: smart-energy-copilot, Property 3: Storage persistence validation**
**Validates: Requirements 3.2**
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.energy_consumption import (
    EnergyConsumption,
    DeviceConsumption, 
    SensorReading,
    BillingPeriod,
    SensorReadings,
    EnergyConsumptionORM,
    SensorReadingORM,
    Base
)
from src.models.repository import EnergyConsumptionRepository, SensorReadingRepository


# Strategy generators for test data (reused from validation tests)

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
    device_count = draw(st.integers(min_value=0, max_value=3))  # Reduced for faster tests
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


class TestDataStorageConsistency:
    """Test data storage persistence validation properties."""
    
    def create_temp_db(self):
        """Create a temporary database for testing."""
        # Create temporary database file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        # Create engine and session
        engine = create_engine(f'sqlite:///{temp_file.name}', echo=False)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        return SessionLocal, temp_file.name, engine
    
    def cleanup_temp_db(self, temp_file_name, engine):
        """Clean up temporary database."""
        engine.dispose()
        if os.path.exists(temp_file_name):
            os.unlink(temp_file_name)
    
    @given(energy_consumption_strategy())
    @settings(deadline=None, max_examples=10)
    def test_energy_consumption_storage_round_trip(self, energy_consumption):
        """Property: For any valid EnergyConsumption, storing then retrieving should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        try:
            with SessionLocal() as db:
                repo = EnergyConsumptionRepository(db)
                
                # Store the energy consumption record
                stored_record = repo.create(energy_consumption)
                
                # Retrieve the record
                retrieved_record = repo.get_by_id(energy_consumption.id)
                
                # Should not be None
                assert retrieved_record is not None
                
                # Convert back to Pydantic model
                retrieved_pydantic = repo.to_pydantic(retrieved_record)
                
                # Should be equivalent to original
                assert retrieved_pydantic == energy_consumption
                assert retrieved_pydantic.model_dump() == energy_consumption.model_dump()
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
    
    @given(sensor_reading_strategy())
    @settings(deadline=None, max_examples=10)
    def test_sensor_reading_storage_round_trip(self, sensor_reading):
        """Property: For any valid SensorReading, storing then retrieving should produce equivalent data."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        try:
            with SessionLocal() as db:
                repo = SensorReadingRepository(db)
                
                # Store the sensor reading
                stored_record = repo.create(sensor_reading)
                
                # Retrieve the latest reading for this sensor
                retrieved_record = repo.get_latest_by_sensor(sensor_reading.sensor_id)
                
                # Should not be None
                assert retrieved_record is not None
                
                # Convert back to Pydantic model
                retrieved_pydantic = repo.to_pydantic(retrieved_record)
                
                # Should be equivalent to original
                assert retrieved_pydantic == sensor_reading
                assert retrieved_pydantic.model_dump() == sensor_reading.model_dump()
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
    
    @given(st.lists(energy_consumption_strategy(), min_size=1, max_size=3))
    @settings(deadline=None, max_examples=5)
    def test_multiple_energy_consumption_storage_consistency(self, energy_consumptions):
        """Property: For any list of EnergyConsumption records, storing all then retrieving should preserve all data."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        # Ensure unique IDs
        unique_consumptions = []
        seen_ids = set()
        for consumption in energy_consumptions:
            if consumption.id not in seen_ids:
                unique_consumptions.append(consumption)
                seen_ids.add(consumption.id)
        
        assume(len(unique_consumptions) > 0)
        
        try:
            with SessionLocal() as db:
                repo = EnergyConsumptionRepository(db)
                
                # Store all records
                stored_records = []
                for consumption in unique_consumptions:
                    stored_record = repo.create(consumption)
                    stored_records.append(stored_record)
                
                # Retrieve all records
                retrieved_records = []
                for consumption in unique_consumptions:
                    retrieved_record = repo.get_by_id(consumption.id)
                    assert retrieved_record is not None
                    retrieved_records.append(retrieved_record)
                
                # Convert to Pydantic models
                retrieved_pydantic = [repo.to_pydantic(record) for record in retrieved_records]
                
                # Should have same number of records
                assert len(retrieved_pydantic) == len(unique_consumptions)
                
                # Each record should match its original
                for original, retrieved in zip(unique_consumptions, retrieved_pydantic):
                    assert retrieved == original
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
    
    @given(st.lists(sensor_reading_strategy(), min_size=1, max_size=3))
    @settings(deadline=None, max_examples=5)
    def test_multiple_sensor_reading_storage_consistency(self, sensor_readings):
        """Property: For any list of SensorReading records, storing all then retrieving should preserve all data."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        assume(len(sensor_readings) > 0)
        
        try:
            with SessionLocal() as db:
                repo = SensorReadingRepository(db)
                
                # Store all records
                stored_records = []
                for reading in sensor_readings:
                    stored_record = repo.create(reading)
                    stored_records.append(stored_record)
                
                # Group by sensor_id for retrieval
                sensor_groups = {}
                for reading in sensor_readings:
                    if reading.sensor_id not in sensor_groups:
                        sensor_groups[reading.sensor_id] = []
                    sensor_groups[reading.sensor_id].append(reading)
                
                # Retrieve and verify each sensor's readings
                for sensor_id, expected_readings in sensor_groups.items():
                    retrieved_records = repo.get_by_sensor_id(sensor_id)
                    
                    # Should have correct number of readings
                    assert len(retrieved_records) == len(expected_readings)
                    
                    # Convert to Pydantic models
                    retrieved_pydantic = [repo.to_pydantic(record) for record in retrieved_records]
                    
                    # Sort both lists by timestamp for comparison
                    expected_sorted = sorted(expected_readings, key=lambda x: x.timestamp)
                    retrieved_sorted = sorted(retrieved_pydantic, key=lambda x: x.timestamp)
                    
                    # Each record should match
                    for expected, retrieved in zip(expected_sorted, retrieved_sorted):
                        assert retrieved == expected
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
    
    @given(energy_consumption_strategy())
    @settings(deadline=None, max_examples=10)
    def test_energy_consumption_update_consistency(self, energy_consumption):
        """Property: For any EnergyConsumption, updating then retrieving should reflect the changes."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        try:
            with SessionLocal() as db:
                repo = EnergyConsumptionRepository(db)
                
                # Store the original record
                stored_record = repo.create(energy_consumption)
                
                # Update some fields
                updates = {
                    'consumption_kwh': energy_consumption.consumption_kwh + 100,
                    'cost_usd': energy_consumption.cost_usd + 50,
                    'confidence_score': min(1.0, energy_consumption.confidence_score + 0.1)
                }
                
                # Apply updates
                updated_record = repo.update(energy_consumption.id, updates)
                
                # Should not be None
                assert updated_record is not None
                
                # Retrieve the updated record
                retrieved_record = repo.get_by_id(energy_consumption.id)
                assert retrieved_record is not None
                
                # Should reflect the updates
                assert retrieved_record.consumption_kwh == updates['consumption_kwh']
                assert retrieved_record.cost_usd == updates['cost_usd']
                assert retrieved_record.confidence_score == updates['confidence_score']
                
                # Other fields should remain unchanged
                assert retrieved_record.id == energy_consumption.id
                assert retrieved_record.source == energy_consumption.source
                assert retrieved_record.timestamp == energy_consumption.timestamp
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
    
    @given(energy_consumption_strategy())
    @settings(deadline=None, max_examples=10)
    def test_energy_consumption_delete_consistency(self, energy_consumption):
        """Property: For any EnergyConsumption, deleting then retrieving should return None."""
        # **Feature: smart-energy-copilot, Property 3: Storage persistence validation**
        # **Validates: Requirements 3.2**
        
        SessionLocal, temp_file_name, engine = self.create_temp_db()
        
        try:
            with SessionLocal() as db:
                repo = EnergyConsumptionRepository(db)
                
                # Store the record
                stored_record = repo.create(energy_consumption)
                
                # Verify it exists
                retrieved_record = repo.get_by_id(energy_consumption.id)
                assert retrieved_record is not None
                
                # Delete the record
                delete_result = repo.delete(energy_consumption.id)
                assert delete_result is True
                
                # Should no longer exist
                retrieved_after_delete = repo.get_by_id(energy_consumption.id)
                assert retrieved_after_delete is None
        finally:
            self.cleanup_temp_db(temp_file_name, engine)
