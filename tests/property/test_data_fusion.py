"""
Property-based tests for multi-source data fusion functionality.

**Validates: Requirements 2.2**
"""

import pytest
import pytest_asyncio
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

from src.services.ai_service import DataFusionEngine, ERNIEModelManager
from src.models.energy_consumption import EnergyConsumption, BillingPeriod
from src.models.sensor_reading import SensorReading


# Test data generators
@st.composite
def energy_consumption_strategy(draw):
    """Generate valid EnergyConsumption instances."""
    timestamp = draw(st.datetimes(
        min_value=datetime(2023, 1, 1),
        max_value=datetime(2024, 12, 31)
    ))
    
    return EnergyConsumption(
        id=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        timestamp=timestamp,
        source=draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry'])),
        consumption_kwh=draw(st.floats(min_value=0.1, max_value=10000.0)),
        cost_usd=draw(st.floats(min_value=0.01, max_value=5000.0)),
        billing_period=BillingPeriod(
            start_date=timestamp - timedelta(days=30),
            end_date=timestamp
        ),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@st.composite
def sensor_reading_strategy(draw):
    """Generate valid SensorReading instances."""
    readings = {}
    
    # Randomly include different sensor types
    if draw(st.booleans()):
        readings['power_watts'] = draw(st.floats(min_value=0, max_value=50000))
    if draw(st.booleans()):
        readings['voltage'] = draw(st.floats(min_value=0, max_value=500))
    if draw(st.booleans()):
        readings['current_amps'] = draw(st.floats(min_value=0, max_value=1000))
    if draw(st.booleans()):
        readings['temperature_celsius'] = draw(st.floats(min_value=-50, max_value=100))
    if draw(st.booleans()):
        readings['humidity_percent'] = draw(st.floats(min_value=0, max_value=100))
    if draw(st.booleans()):
        readings['occupancy'] = draw(st.booleans())
    
    return SensorReading(
        sensor_id=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        device_type=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        timestamp=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2024, 12, 31)
        )),
        readings=readings,
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        location=draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    )


class TestDataFusion:
    """Property-based tests for data fusion functionality."""
    
    @pytest_asyncio.fixture
    async def fusion_engine(self):
        """Create a DataFusionEngine instance for testing."""
        model_manager = ERNIEModelManager()
        await model_manager.load_model()
        return DataFusionEngine(model_manager)
    
    @given(
        utility_data=st.lists(energy_consumption_strategy(), min_size=1, max_size=10),
        sensor_data=st.lists(sensor_reading_strategy(), min_size=1, max_size=10)
    )
    @settings(max_examples=5, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_multi_source_data_combination(self, fusion_engine, utility_data, sensor_data):
        """
        Property 11: Multi-source data combination
        
        For any combination of utility bill data and IoT sensor data,
        the fusion engine should successfully combine them into a unified dataset
        with preserved data integrity and calculated quality scores.
        
        **Validates: Requirements 2.2**
        """
        # Execute data fusion
        result = await fusion_engine.fuse_multi_source_data(
            utility_data=utility_data,
            sensor_data=sensor_data
        )
        
        # Property: Result should contain all expected fields
        assert 'utility_consumption' in result
        assert 'sensor_readings' in result
        assert 'combined_insights' in result
        assert 'data_quality_score' in result
        assert 'fusion_timestamp' in result
        
        # Property: All utility data should be preserved
        assert len(result['utility_consumption']) == len(utility_data)
        for i, original in enumerate(utility_data):
            fused_item = result['utility_consumption'][i]
            assert fused_item['timestamp'] == original.timestamp
            assert fused_item['consumption_kwh'] == original.consumption_kwh
            assert fused_item['cost_usd'] == original.cost_usd
            assert fused_item['source'] == original.source
            assert fused_item['confidence'] == original.confidence_score
        
        # Property: All sensor data should be preserved
        assert len(result['sensor_readings']) == len(sensor_data)
        for i, original in enumerate(sensor_data):
            fused_item = result['sensor_readings'][i]
            assert fused_item['timestamp'] == original.timestamp
            assert fused_item['sensor_id'] == original.sensor_id
            assert fused_item['device_type'] == original.device_type
            assert fused_item['readings'] == original.readings
            assert fused_item['quality_score'] == original.quality_score
            assert fused_item['location'] == original.location
        
        # Property: Data quality score should be valid
        assert 0.0 <= result['data_quality_score'] <= 1.0
        
        # Property: Combined insights should be generated
        assert isinstance(result['combined_insights'], list)
        
        # Property: Fusion timestamp should be recent
        time_diff = datetime.now() - result['fusion_timestamp']
        assert time_diff.total_seconds() < 60  # Within last minute
    
    @given(
        utility_data=st.lists(energy_consumption_strategy(), min_size=0, max_size=5),
        sensor_data=st.lists(sensor_reading_strategy(), min_size=0, max_size=5)
    )
    @settings(max_examples=5, deadline=3000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, fusion_engine, utility_data, sensor_data):
        """
        Property: Data fusion should handle empty datasets gracefully.
        
        For any combination including empty datasets, the fusion should complete
        successfully and return appropriate default values.
        """
        # Skip if both datasets are empty (not a valid test case)
        if not utility_data and not sensor_data:
            return
        
        result = await fusion_engine.fuse_multi_source_data(
            utility_data=utility_data,
            sensor_data=sensor_data
        )
        
        # Property: Result structure should be consistent regardless of empty inputs
        assert 'utility_consumption' in result
        assert 'sensor_readings' in result
        assert 'combined_insights' in result
        assert 'data_quality_score' in result
        
        # Property: Empty lists should be preserved
        assert len(result['utility_consumption']) == len(utility_data)
        assert len(result['sensor_readings']) == len(sensor_data)
        
        # Property: Quality score should be 0 if no data, otherwise valid
        if not utility_data and not sensor_data:
            assert result['data_quality_score'] == 0.0
        else:
            assert 0.0 <= result['data_quality_score'] <= 1.0
    
    @given(
        high_quality_data=st.lists(
            energy_consumption_strategy().map(
                lambda x: x.model_copy(update={'confidence_score': 0.9})
            ),
            min_size=1, max_size=5
        ),
        low_quality_data=st.lists(
            energy_consumption_strategy().map(
                lambda x: x.model_copy(update={'confidence_score': 0.3})
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=5, deadline=3000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, fusion_engine, high_quality_data, low_quality_data):
        """
        Property: Data quality score should reflect the quality of input data.
        
        High quality data should result in higher fusion quality scores
        than low quality data.
        """
        # Test high quality data
        high_result = await fusion_engine.fuse_multi_source_data(
            utility_data=high_quality_data,
            sensor_data=[]
        )
        
        # Test low quality data
        low_result = await fusion_engine.fuse_multi_source_data(
            utility_data=low_quality_data,
            sensor_data=[]
        )
        
        # Property: High quality data should have higher quality score
        assert high_result['data_quality_score'] > low_result['data_quality_score']
        
        # Property: Both scores should be valid
        assert 0.0 <= high_result['data_quality_score'] <= 1.0
        assert 0.0 <= low_result['data_quality_score'] <= 1.0
    
    @given(
        utility_data=st.lists(energy_consumption_strategy(), min_size=1, max_size=3),
        sensor_data=st.lists(sensor_reading_strategy(), min_size=1, max_size=3)
    )
    @settings(max_examples=5, deadline=3000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_insight_generation(self, fusion_engine, utility_data, sensor_data):
        """
        Property: When both utility and sensor data are present,
        correlation insights should be generated.
        """
        result = await fusion_engine.fuse_multi_source_data(
            utility_data=utility_data,
            sensor_data=sensor_data
        )
        
        # Property: Insights should be generated when both data types are present
        insights = result['combined_insights']
        assert isinstance(insights, list)
        
        # Property: At least one correlation insight should be present
        correlation_insights = [
            insight for insight in insights 
            if insight.get('insight_type') == 'correlation'
        ]
        assert len(correlation_insights) >= 1
        
        # Property: Each insight should have required fields
        for insight in insights:
            assert 'insight_type' in insight
            assert 'title' in insight
            assert 'description' in insight
            assert 'confidence' in insight
            assert 'data_sources' in insight
            assert 'timestamp' in insight
            
            # Property: Confidence should be valid
            assert 0.0 <= insight['confidence'] <= 1.0
    
    @given(
        data_batch=st.lists(energy_consumption_strategy(), min_size=2, max_size=10)
    )
    @settings(max_examples=5, deadline=3000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_fusion_idempotency(self, fusion_engine, data_batch):
        """
        Property: Fusing the same data multiple times should produce
        consistent results (excluding timestamp fields).
        """
        # Perform fusion twice
        result1 = await fusion_engine.fuse_multi_source_data(
            utility_data=data_batch,
            sensor_data=[]
        )
        
        result2 = await fusion_engine.fuse_multi_source_data(
            utility_data=data_batch,
            sensor_data=[]
        )
        
        # Property: Core data should be identical (excluding timestamps)
        assert len(result1['utility_consumption']) == len(result2['utility_consumption'])
        assert result1['data_quality_score'] == result2['data_quality_score']
        
        # Property: Utility consumption data should be identical
        for item1, item2 in zip(result1['utility_consumption'], result2['utility_consumption']):
            assert item1['consumption_kwh'] == item2['consumption_kwh']
            assert item1['cost_usd'] == item2['cost_usd']
            assert item1['source'] == item2['source']
            assert item1['confidence'] == item2['confidence']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
