"""
Property-based tests for energy pattern analysis functionality.

**Feature: smart-energy-copilot, Property 10: Energy pattern identification**
**Validates: Requirements 2.1**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
import asyncio
from typing import List

from src.services.ai_service import EnergyPatternAnalyzer, ERNIEModelManager, EnergyPattern
from src.models.energy_consumption import EnergyConsumption


class MockERNIEModelManager:
    """Mock ERNIE model manager for testing."""
    
    def __init__(self):
        self._model_loaded = True
    
    def is_loaded(self) -> bool:
        return self._model_loaded
    
    async def load_model(self) -> bool:
        return True
    
    async def analyze_energy_text(self, text: str) -> dict:
        return {
            "energy_relevance": 0.8,
            "sentiment": "neutral",
            "key_concepts": ["energy", "consumption"],
            "confidence": 0.85
        }


# Strategy for generating valid energy consumption data
@st.composite
def energy_consumption_strategy(draw):
    """Generate valid EnergyConsumption objects."""
    base_time = datetime(2024, 1, 1)
    hours_offset = draw(st.integers(min_value=0, max_value=8760))  # Up to 1 year
    timestamp = base_time + timedelta(hours=hours_offset)
    
    consumption_kwh = draw(st.floats(min_value=0.1, max_value=1000.0))
    cost_usd = draw(st.floats(min_value=0.01, max_value=500.0))
    source = draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry']))
    confidence_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return EnergyConsumption(
        id=f"test_{hours_offset}",
        timestamp=timestamp,
        source=source,
        consumption_kwh=consumption_kwh,
        cost_usd=cost_usd,
        billing_period={
            'start_date': timestamp - timedelta(days=30),
            'end_date': timestamp
        },
        confidence_score=confidence_score
    )


@st.composite
def energy_consumption_list_strategy(draw):
    """Generate lists of energy consumption data."""
    size = draw(st.integers(min_value=1, max_value=100))
    return draw(st.lists(energy_consumption_strategy(), min_size=size, max_size=size))


class TestEnergyPatternAnalysis:
    """Test energy pattern identification properties."""
    
    def _create_pattern_analyzer(self):
        """Create a pattern analyzer with mock model manager."""
        mock_manager = MockERNIEModelManager()
        return EnergyPatternAnalyzer(mock_manager)
    
    @given(consumption_data=energy_consumption_list_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_pattern_identification_always_returns_list(self, consumption_data):
        """
        Property: Pattern identification always returns a list of patterns.
        
        For any valid energy consumption data, the pattern analyzer should
        always return a list (possibly empty) of EnergyPattern objects.
        """
        pattern_analyzer = self._create_pattern_analyzer()
        
        async def run_test():
            patterns = await pattern_analyzer.identify_patterns(consumption_data)
            
            # Should always return a list
            assert isinstance(patterns, list)
            
            # All items in the list should be EnergyPattern objects
            for pattern in patterns:
                assert isinstance(pattern, EnergyPattern)
                
            # Pattern types should be valid
            valid_types = {'daily', 'weekly', 'seasonal', 'anomaly'}
            for pattern in patterns:
                assert pattern.pattern_type in valid_types
                
            # Confidence scores should be between 0 and 1
            for pattern in patterns:
                assert 0.0 <= pattern.confidence <= 1.0
                
            # Time ranges should be valid
            for pattern in patterns:
                start_time, end_time = pattern.time_range
                assert isinstance(start_time, datetime)
                assert isinstance(end_time, datetime)
                assert start_time <= end_time
        
        # Run the async test
        asyncio.run(run_test())
    
    @given(consumption_data=energy_consumption_list_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_pattern_identification_deterministic(self, consumption_data):
        """
        Property: Pattern identification is deterministic for the same input.
        
        For any given energy consumption data, running pattern identification
        multiple times should produce the same results.
        """
        pattern_analyzer = self._create_pattern_analyzer()
        
        async def run_test():
            # Run pattern identification twice
            patterns1 = await pattern_analyzer.identify_patterns(consumption_data)
            patterns2 = await pattern_analyzer.identify_patterns(consumption_data)
            
            # Should return the same number of patterns
            assert len(patterns1) == len(patterns2)
            
            # Patterns should have the same types and confidence scores
            for p1, p2 in zip(patterns1, patterns2):
                assert p1.pattern_type == p2.pattern_type
                assert abs(p1.confidence - p2.confidence) < 0.001  # Allow for floating point precision
                assert p1.consumption_trend == p2.consumption_trend
        
        asyncio.run(run_test())
    
    @given(
        consumption_data=energy_consumption_list_strategy(),
        additional_data=energy_consumption_list_strategy()
    )
    @settings(max_examples=20, deadline=5000)
    def test_pattern_identification_monotonic_with_more_data(self, consumption_data, additional_data):
        """
        Property: More data should not decrease pattern identification capability.
        
        For any energy consumption data, adding more valid data points should
        not result in fewer patterns being identified (monotonic property).
        """
        assume(len(consumption_data) >= 5)  # Need minimum data for meaningful patterns
        assume(len(additional_data) >= 1)
        
        pattern_analyzer = self._create_pattern_analyzer()
        
        async def run_test():
            # Identify patterns with original data
            patterns_original = await pattern_analyzer.identify_patterns(consumption_data)
            
            # Identify patterns with combined data
            combined_data = consumption_data + additional_data
            patterns_combined = await pattern_analyzer.identify_patterns(combined_data)
            
            # With more data, we should identify at least as many patterns
            # (or the same number with higher confidence)
            assert len(patterns_combined) >= len(patterns_original)
            
            # If same number of patterns, confidence should not decrease significantly
            if len(patterns_combined) == len(patterns_original):
                for orig, combined in zip(patterns_original, patterns_combined):
                    if orig.pattern_type == combined.pattern_type:
                        # Confidence should not decrease by more than 10%
                        assert combined.confidence >= orig.confidence - 0.1
        
        asyncio.run(run_test())
    
    @given(st.lists(energy_consumption_strategy(), min_size=0, max_size=0))
    @settings(max_examples=10)
    def test_pattern_identification_empty_data(self, pattern_analyzer, consumption_data):
        """
        Property: Empty data should return empty pattern list.
        
        When no consumption data is provided, pattern identification
        should return an empty list without errors.
        """
        async def run_test():
            patterns = await pattern_analyzer.identify_patterns(consumption_data)
            assert isinstance(patterns, list)
            assert len(patterns) == 0
        
        asyncio.run(run_test())
    
    @given(consumption_data=st.lists(energy_consumption_strategy(), min_size=1, max_size=5))
    @settings(max_examples=20, deadline=3000)
    def test_pattern_identification_minimal_data(self, pattern_analyzer, consumption_data):
        """
        Property: Minimal data should handle gracefully.
        
        With very little data (1-5 points), pattern identification should
        still work without errors, though it may return fewer patterns.
        """
        async def run_test():
            patterns = await pattern_analyzer.identify_patterns(consumption_data)
            
            # Should not crash and should return a list
            assert isinstance(patterns, list)
            
            # With minimal data, we expect fewer patterns
            assert len(patterns) <= 2  # At most daily and anomaly patterns
            
            # All returned patterns should be valid
            for pattern in patterns:
                assert isinstance(pattern, EnergyPattern)
                assert 0.0 <= pattern.confidence <= 1.0
        
        asyncio.run(run_test())
    
    @given(
        consumption_data=st.lists(
            energy_consumption_strategy(), 
            min_size=24, 
            max_size=24
        )
    )
    @settings(max_examples=15, deadline=4000)
    def test_daily_pattern_identification(self, pattern_analyzer, consumption_data):
        """
        Property: With 24+ hours of data, daily patterns should be identifiable.
        
        When sufficient hourly data is available, the system should be able
        to identify daily consumption patterns.
        """
        async def run_test():
            patterns = await pattern_analyzer.identify_patterns(consumption_data)
            
            # Should identify at least one pattern with sufficient data
            assert len(patterns) >= 1
            
            # Should include daily pattern analysis
            pattern_types = [p.pattern_type for p in patterns]
            assert 'daily' in pattern_types or 'anomaly' in pattern_types
            
            # Daily patterns should have valid peak hours
            daily_patterns = [p for p in patterns if p.pattern_type == 'daily']
            for pattern in daily_patterns:
                assert isinstance(pattern.peak_hours, list)
                # Peak hours should be valid (0-23)
                for hour in pattern.peak_hours:
                    assert 0 <= hour <= 23
        
        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])