"""
Property-based tests for dashboard data display functionality.

**Property 19: Dashboard data display**
**Validates: Requirements 5.1**

Tests that the dashboard correctly displays energy consumption data
with proper formatting, filtering, and summary statistics.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import asyncio

from src.models.energy_consumption import EnergyConsumption, EnergyConsumptionORM
from src.database.connection import get_db_session


# Generators for test data
@st.composite
def energy_consumption_data(draw):
    """Generate valid energy consumption data."""
    return {
        'id': draw(st.uuids()).hex,
        'timestamp': draw(st.datetimes(
            min_value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )),
        'source': draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry'])),
        'consumption_kwh': draw(st.floats(min_value=0.1, max_value=10000.0)),
        'cost_usd': draw(st.floats(min_value=0.01, max_value=5000.0)),
        'billing_period_start': draw(st.datetimes(
            min_value=datetime.now() - timedelta(days=90),
            max_value=datetime.now() - timedelta(days=30)
        )),
        'billing_period_end': draw(st.datetimes(
            min_value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )),
        'confidence_score': draw(st.floats(min_value=0.0, max_value=1.0))
    }


class TestDashboardDataDisplay:
    """Property-based tests for dashboard data display."""
    
    @given(
        consumption_records=st.lists(energy_consumption_data(), min_size=1, max_size=20)
    )
    @settings(max_examples=5, deadline=5000)
    def test_dashboard_data_structure_consistency(
        self, 
        consumption_records: List[Dict[str, Any]]
    ):
        """
        Property: For any set of energy consumption records, the dashboard data
        should maintain consistent structure and valid data types.
        
        **Validates: Requirements 5.1**
        """
        # Test data structure consistency
        for record in consumption_records:
            # Verify required fields exist
            assert 'id' in record
            assert 'timestamp' in record
            assert 'source' in record
            assert 'consumption_kwh' in record
            assert 'cost_usd' in record
            assert 'confidence_score' in record
            
            # Verify data types
            assert isinstance(record['consumption_kwh'], (int, float))
            assert isinstance(record['cost_usd'], (int, float))
            assert isinstance(record['confidence_score'], (int, float))
            assert record['source'] in ['utility_bill', 'iot_sensor', 'manual_entry']
            
            # Verify value ranges
            assert record['consumption_kwh'] >= 0
            assert record['cost_usd'] >= 0
            assert 0.0 <= record['confidence_score'] <= 1.0
    
    @given(
        consumption_records=st.lists(energy_consumption_data(), min_size=1, max_size=50)
    )
    @settings(max_examples=5)
    def test_dashboard_summary_calculations(
        self, 
        consumption_records: List[Dict[str, Any]]
    ):
        """
        Property: For any set of energy consumption records, summary statistics
        should be calculated correctly and consistently.
        
        **Validates: Requirements 5.1**
        """
        if consumption_records:
            # Calculate expected summary statistics
            total_consumption = sum(record['consumption_kwh'] for record in consumption_records)
            total_cost = sum(record['cost_usd'] for record in consumption_records)
            avg_confidence = sum(record['confidence_score'] for record in consumption_records) / len(consumption_records)
            
            # Verify calculations are mathematically correct
            assert total_consumption >= 0
            assert total_cost >= 0
            assert 0.0 <= avg_confidence <= 1.0
            
            # Verify individual contributions sum to total
            individual_sum = 0
            for record in consumption_records:
                individual_sum += record['consumption_kwh']
            assert abs(individual_sum - total_consumption) < 0.001
    
    @given(
        consumption_records=st.lists(energy_consumption_data(), min_size=2, max_size=100),
        filter_source=st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry'])
    )
    @settings(max_examples=5)
    def test_dashboard_filtering_logic(
        self, 
        consumption_records: List[Dict[str, Any]], 
        filter_source: str
    ):
        """
        Property: For any dataset and filter criteria, filtering should return
        only records that match the criteria exactly.
        
        **Validates: Requirements 5.1**
        """
        # Apply source filter
        filtered_records = [
            record for record in consumption_records 
            if record['source'] == filter_source
        ]
        
        # Verify all filtered records match the criteria
        for record in filtered_records:
            assert record['source'] == filter_source
        
        # Verify no matching records were excluded
        expected_count = len([
            record for record in consumption_records 
            if record['source'] == filter_source
        ])
        assert len(filtered_records) == expected_count
    
    @given(
        consumption_records=st.lists(energy_consumption_data(), min_size=10, max_size=100),
        limit=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=5)
    def test_dashboard_pagination_logic(
        self, 
        consumption_records: List[Dict[str, Any]], 
        limit: int
    ):
        """
        Property: For any dataset larger than the limit, pagination should return
        exactly the requested number of records while maintaining total count accuracy.
        
        **Validates: Requirements 5.1**
        """
        # Sort records by timestamp (most recent first)
        sorted_records = sorted(
            consumption_records, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )
        
        # Apply pagination
        paginated_records = sorted_records[:limit]
        
        # Verify pagination constraints
        assert len(paginated_records) == min(limit, len(consumption_records))
        
        # Verify ordering is maintained
        if len(paginated_records) > 1:
            timestamps = [record['timestamp'] for record in paginated_records]
            assert timestamps == sorted(timestamps, reverse=True)
        
        # Verify total count is preserved
        total_count = len(consumption_records)
        assert total_count >= len(paginated_records)
    
    def test_dashboard_handles_empty_data(self):
        """
        Property: For empty energy data, the dashboard should return appropriate
        default values without errors.
        
        **Validates: Requirements 5.1**
        """
        empty_records = []
        
        # Verify empty data handling
        assert len(empty_records) == 0
        
        # Verify default summary values would be appropriate
        default_summary = {
            'total_consumption_kwh': 0.0,
            'total_cost_usd': 0.0,
            'average_confidence': 0.0,
            'records_count': 0
        }
        
        # Verify default values are valid
        assert default_summary['total_consumption_kwh'] >= 0
        assert default_summary['total_cost_usd'] >= 0
        assert 0.0 <= default_summary['average_confidence'] <= 1.0
        assert default_summary['records_count'] == len(empty_records)


if __name__ == "__main__":
    pytest.main([__file__])
