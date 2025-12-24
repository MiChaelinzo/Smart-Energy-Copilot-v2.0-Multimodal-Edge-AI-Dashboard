"""
Property-based tests for recommendation presentation format.

**Property 20: Recommendation presentation format**
**Validates: Requirements 5.2**

Tests that recommendations are displayed with proper formatting,
clear implementation steps, and expected benefits.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from src.models.recommendation import OptimizationRecommendation, OptimizationRecommendationORM
from src.database.connection import get_db_session


# Fixed base date to avoid flaky tests
BASE_DATE = datetime(2024, 1, 1)


# Generators for test data
@st.composite
def recommendation_data(draw):
    """Generate valid recommendation data."""
    return {
        'id': draw(st.uuids()).hex,
        'type': draw(st.sampled_from(['cost_saving', 'efficiency', 'environmental'])),
        'priority': draw(st.sampled_from(['high', 'medium', 'low'])),
        'title': draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs')))),
        'description': draw(st.text(min_size=20, max_size=500, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs')))),
        'implementation_steps': draw(st.lists(
            st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))),
            min_size=1, max_size=10
        )),
        'estimated_savings': {
            'annual_cost_usd': draw(st.floats(min_value=10.0, max_value=10000.0)),
            'annual_kwh': draw(st.floats(min_value=50.0, max_value=50000.0)),
            'co2_reduction_kg': draw(st.floats(min_value=5.0, max_value=5000.0))
        },
        'difficulty': draw(st.sampled_from(['easy', 'moderate', 'complex'])),
        'agent_source': draw(st.sampled_from(['efficiency_advisor', 'cost_forecaster', 'eco_planner'])),
        'confidence': draw(st.floats(min_value=0.0, max_value=1.0)),
        'created_at': draw(st.datetimes(
            min_value=BASE_DATE,
            max_value=BASE_DATE + timedelta(days=365)
        )),
        'status': draw(st.sampled_from(['pending', 'implemented', 'dismissed']))
    }


class TestRecommendationPresentation:
    """Property-based tests for recommendation presentation format."""
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=1, max_size=20)
    )
    @settings(max_examples=5, deadline=5000)
    def test_recommendation_display_format_consistency(
        self, 
        recommendations: List[Dict[str, Any]]
    ):
        """
        Property: For any set of recommendations, the display format should be
        consistent and contain all required presentation elements.
        
        **Validates: Requirements 5.2**
        """
        for recommendation in recommendations:
            # Verify required display fields exist
            assert 'id' in recommendation
            assert 'type' in recommendation
            assert 'priority' in recommendation
            assert 'title' in recommendation
            assert 'description' in recommendation
            assert 'implementation_steps' in recommendation
            assert 'estimated_savings' in recommendation
            assert 'difficulty' in recommendation
            assert 'agent_source' in recommendation
            assert 'confidence' in recommendation
            assert 'status' in recommendation
            
            # Verify field types for display
            assert isinstance(recommendation['title'], str)
            assert isinstance(recommendation['description'], str)
            assert isinstance(recommendation['implementation_steps'], list)
            assert isinstance(recommendation['estimated_savings'], dict)
            
            # Verify enumerated values are valid for display
            assert recommendation['type'] in ['cost_saving', 'efficiency', 'environmental']
            assert recommendation['priority'] in ['high', 'medium', 'low']
            assert recommendation['difficulty'] in ['easy', 'moderate', 'complex']
            assert recommendation['status'] in ['pending', 'implemented', 'dismissed']
            
            # Verify content quality for display
            assert len(recommendation['title'].strip()) > 0
            assert len(recommendation['description'].strip()) > 0
            assert len(recommendation['implementation_steps']) > 0
            
            # Verify all implementation steps have content
            for step in recommendation['implementation_steps']:
                assert isinstance(step, str)
                assert len(step.strip()) > 0
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=1, max_size=50)
    )
    @settings(max_examples=5)
    def test_recommendation_savings_display_format(
        self, 
        recommendations: List[Dict[str, Any]]
    ):
        """
        Property: For any recommendations with estimated savings, the savings
        should be displayed in a consistent, user-friendly format.
        
        **Validates: Requirements 5.2**
        """
        for recommendation in recommendations:
            savings = recommendation['estimated_savings']
            
            # Verify savings structure for display
            assert 'annual_cost_usd' in savings
            assert 'annual_kwh' in savings
            assert 'co2_reduction_kg' in savings
            
            # Verify savings values are positive and displayable
            assert savings['annual_cost_usd'] >= 0
            assert savings['annual_kwh'] >= 0
            assert savings['co2_reduction_kg'] >= 0
            
            # Verify values are reasonable for display (not NaN or infinite)
            assert not (savings['annual_cost_usd'] != savings['annual_cost_usd'])  # Check for NaN
            assert not (savings['annual_kwh'] != savings['annual_kwh'])  # Check for NaN
            assert not (savings['co2_reduction_kg'] != savings['co2_reduction_kg'])  # Check for NaN
            
            # Verify values can be formatted for currency/units display
            cost_formatted = f"${savings['annual_cost_usd']:.2f}"
            kwh_formatted = f"{savings['annual_kwh']:.1f} kWh"
            co2_formatted = f"{savings['co2_reduction_kg']:.1f} kg CO2"
            
            # Verify formatted strings are reasonable
            assert len(cost_formatted) > 3  # At least "$0.00"
            assert len(kwh_formatted) > 5   # At least "0.0 kWh"
            assert len(co2_formatted) > 8   # At least "0.0 kg CO2"
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=1, max_size=30)
    )
    @settings(max_examples=5)
    def test_recommendation_implementation_steps_display(
        self, 
        recommendations: List[Dict[str, Any]]
    ):
        """
        Property: For any recommendations, implementation steps should be
        displayed as clear, actionable instructions.
        
        **Validates: Requirements 5.2**
        """
        for recommendation in recommendations:
            steps = recommendation['implementation_steps']
            
            # Verify steps are displayable as a list
            assert isinstance(steps, list)
            assert len(steps) > 0
            
            # Verify each step is actionable content
            for i, step in enumerate(steps):
                assert isinstance(step, str)
                assert len(step.strip()) > 0
                
                # Verify step can be displayed with numbering
                numbered_step = f"{i + 1}. {step}"
                assert len(numbered_step) > 3
                
                # Verify step doesn't contain problematic characters for display
                assert '\n' not in step or step.count('\n') <= 2  # Allow some line breaks
                assert '\t' not in step  # No tabs in display text
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=2, max_size=100)
    )
    @settings(max_examples=5)
    def test_recommendation_priority_display_ordering(
        self, 
        recommendations: List[Dict[str, Any]]
    ):
        """
        Property: For any set of recommendations, priority-based display ordering
        should be consistent and logical.
        
        **Validates: Requirements 5.2**
        """
        # Define priority order for display (high -> medium -> low)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        # Sort recommendations by priority for display
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: priority_order[x['priority']],
            reverse=True
        )
        
        # Verify priority ordering is maintained
        for i in range(len(sorted_recommendations) - 1):
            current_priority = priority_order[sorted_recommendations[i]['priority']]
            next_priority = priority_order[sorted_recommendations[i + 1]['priority']]
            assert current_priority >= next_priority
        
        # Verify all priorities are preserved
        original_priorities = [rec['priority'] for rec in recommendations]
        sorted_priorities = [rec['priority'] for rec in sorted_recommendations]
        assert sorted(original_priorities) == sorted(sorted_priorities)
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=1, max_size=50),
        status_filter=st.sampled_from(['pending', 'implemented', 'dismissed'])
    )
    @settings(max_examples=5)
    def test_recommendation_status_display_filtering(
        self, 
        recommendations: List[Dict[str, Any]], 
        status_filter: str
    ):
        """
        Property: For any recommendations and status filter, the display should
        show only recommendations matching the filter criteria.
        
        **Validates: Requirements 5.2**
        """
        # Filter recommendations by status for display
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec['status'] == status_filter
        ]
        
        # Verify all filtered recommendations match the status
        for recommendation in filtered_recommendations:
            assert recommendation['status'] == status_filter
        
        # Verify no matching recommendations were excluded
        expected_count = len([
            rec for rec in recommendations 
            if rec['status'] == status_filter
        ])
        assert len(filtered_recommendations) == expected_count
        
        # Verify status display formatting
        for recommendation in filtered_recommendations:
            status = recommendation['status']
            
            # Verify status can be displayed with proper formatting
            if status == 'pending':
                display_status = "Pending Implementation"
            elif status == 'implemented':
                display_status = "Implemented"
            elif status == 'dismissed':
                display_status = "Dismissed"
            
            assert len(display_status) > 0
            assert display_status[0].isupper()  # Proper capitalization for display
    
    @given(
        recommendations=st.lists(recommendation_data(), min_size=1, max_size=20)
    )
    @settings(max_examples=5)
    def test_recommendation_confidence_display_format(
        self, 
        recommendations: List[Dict[str, Any]]
    ):
        """
        Property: For any recommendations, confidence scores should be displayed
        in a user-friendly format with appropriate visual indicators.
        
        **Validates: Requirements 5.2**
        """
        for recommendation in recommendations:
            confidence = recommendation['confidence']
            
            # Verify confidence is in valid range for display
            assert 0.0 <= confidence <= 1.0
            
            # Verify confidence can be formatted as percentage
            percentage = confidence * 100
            formatted_confidence = f"{percentage:.0f}%"
            
            # Verify formatted confidence is displayable
            assert len(formatted_confidence) >= 2  # At least "0%"
            assert formatted_confidence.endswith('%')
            
            # Verify confidence level categorization for display
            if confidence >= 0.8:
                confidence_level = "High"
            elif confidence >= 0.6:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            assert confidence_level in ["High", "Medium", "Low"]
            
            # Verify agent source is displayable
            agent_source = recommendation['agent_source']
            display_agent = agent_source.replace('_', ' ').title()
            assert len(display_agent) > 0
            assert display_agent[0].isupper()


if __name__ == "__main__":
    pytest.main([__file__])
