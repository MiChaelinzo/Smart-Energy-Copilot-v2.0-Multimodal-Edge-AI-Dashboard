"""
Property-based tests for recommendation prioritization functionality.

**Validates: Requirements 2.4**
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
from typing import List, Dict, Any

from src.services.recommendation_engine import RecommendationPrioritizer
from src.models.recommendation import OptimizationRecommendation, EstimatedSavings


# Test data generators
@st.composite
def estimated_savings_strategy(draw):
    """Generate valid EstimatedSavings instances."""
    return EstimatedSavings(
        annual_cost_usd=draw(st.floats(min_value=0.0, max_value=5000.0)),
        annual_kwh=draw(st.floats(min_value=0.0, max_value=20000.0)),
        co2_reduction_kg=draw(st.floats(min_value=0.0, max_value=10000.0))
    )


@st.composite
def optimization_recommendation_strategy(draw):
    """Generate valid OptimizationRecommendation instances."""
    return OptimizationRecommendation(
        id=draw(st.text(min_size=1, max_size=50)),
        type=draw(st.sampled_from(['cost_saving', 'efficiency', 'environmental'])),
        priority=draw(st.sampled_from(['high', 'medium', 'low'])),
        title=draw(st.text(min_size=5, max_size=100)),
        description=draw(st.text(min_size=10, max_size=500)),
        implementation_steps=draw(st.lists(st.text(min_size=5, max_size=100), min_size=1, max_size=5)),
        estimated_savings=draw(estimated_savings_strategy()),
        difficulty=draw(st.sampled_from(['easy', 'moderate', 'complex'])),
        agent_source=draw(st.text(min_size=1, max_size=50)),
        confidence=draw(st.floats(min_value=0.1, max_value=1.0)),
        created_at=draw(st.datetimes(
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31)
        )),
        status='pending'
    )


class TestRecommendationPrioritization:
    """Property-based tests for recommendation prioritization functionality."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create a RecommendationPrioritizer instance for testing."""
        return RecommendationPrioritizer()
    
    @given(recommendations=st.lists(optimization_recommendation_strategy(), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=5000)
    @pytest.mark.asyncio
    async def test_recommendation_ranking_algorithm(self, prioritizer, recommendations):
        """
        Property 17: Recommendation ranking algorithm
        
        For any list of recommendations, the prioritization algorithm should
        rank them consistently based on impact, difficulty, and other factors,
        with higher-impact, easier recommendations receiving higher priority.
        
        **Validates: Requirements 2.4**
        """
        # Prioritize recommendations
        prioritized = await prioritizer.prioritize_recommendations(recommendations)
        
        # Property: Should return same number of recommendations
        assert len(prioritized) == len(recommendations)
        
        # Property: All original recommendations should be present
        original_ids = {rec.id for rec in recommendations}
        prioritized_ids = {rec.id for rec in prioritized}
        assert original_ids == prioritized_ids
        
        # Property: Priority field should be updated appropriately
        high_priority = [r for r in prioritized if r.priority == 'high']
        medium_priority = [r for r in prioritized if r.priority == 'medium']
        low_priority = [r for r in prioritized if r.priority == 'low']
        
        # Property: Should have some distribution of priorities (unless all identical)
        total_recs = len(prioritized)
        if total_recs > 1:
            # At least one priority level should be assigned
            assert len(high_priority) + len(medium_priority) + len(low_priority) == total_recs
        
        # Property: High priority recommendations should generally have better characteristics
        if high_priority and low_priority:
            # Compare average savings between high and low priority
            high_avg_savings = sum(r.estimated_savings.annual_cost_usd for r in high_priority) / len(high_priority)
            low_avg_savings = sum(r.estimated_savings.annual_cost_usd for r in low_priority) / len(low_priority)
            
            # High priority should generally have higher savings (allow some flexibility)
            assert high_avg_savings >= low_avg_savings * 0.7  # At least 70% of low priority savings
        
        # Property: Recommendations should be ordered by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        for i in range(len(prioritized) - 1):
            current_priority = priority_order[prioritized[i].priority]
            next_priority = priority_order[prioritized[i + 1].priority]
            assert current_priority >= next_priority, "Recommendations should be ordered by priority"
    
    @given(
        high_savings_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={
                    'estimated_savings': EstimatedSavings(
                        annual_cost_usd=2000.0,
                        annual_kwh=8000.0,
                        co2_reduction_kg=3000.0
                    ),
                    'difficulty': 'easy'
                })
            ),
            min_size=1, max_size=3
        ),
        low_savings_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={
                    'estimated_savings': EstimatedSavings(
                        annual_cost_usd=50.0,
                        annual_kwh=200.0,
                        co2_reduction_kg=100.0
                    ),
                    'difficulty': 'complex'
                })
            ),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.asyncio
    async def test_impact_based_prioritization(self, prioritizer, high_savings_recs, low_savings_recs):
        """
        Property: High-impact, easy recommendations should be prioritized
        over low-impact, complex recommendations.
        """
        all_recommendations = high_savings_recs + low_savings_recs
        prioritized = await prioritizer.prioritize_recommendations(all_recommendations)
        
        # Property: High savings, easy recommendations should appear first
        high_savings_ids = {rec.id for rec in high_savings_recs}
        low_savings_ids = {rec.id for rec in low_savings_recs}
        
        # Find positions of high and low savings recommendations
        high_positions = [i for i, rec in enumerate(prioritized) if rec.id in high_savings_ids]
        low_positions = [i for i, rec in enumerate(prioritized) if rec.id in low_savings_ids]
        
        if high_positions and low_positions:
            # Average position of high savings should be better (lower index) than low savings
            avg_high_pos = sum(high_positions) / len(high_positions)
            avg_low_pos = sum(low_positions) / len(low_positions)
            
            assert avg_high_pos <= avg_low_pos, "High-impact recommendations should be prioritized higher"
    
    @given(
        recommendations=st.lists(optimization_recommendation_strategy(), min_size=2, max_size=8),
        user_preferences=st.dictionaries(
            st.sampled_from(['prioritize_cost', 'prioritize_environment', 'prefer_easy_implementation']),
            st.booleans(),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=20, deadline=6000)
    @pytest.mark.asyncio
    async def test_user_preference_influence(self, prioritizer, recommendations, user_preferences):
        """
        Property: User preferences should influence prioritization.
        
        Different user preferences should lead to different prioritization
        of the same set of recommendations.
        """
        # Prioritize with user preferences
        prioritized_with_prefs = await prioritizer.prioritize_recommendations(
            recommendations, user_preferences
        )
        
        # Prioritize without user preferences
        prioritized_without_prefs = await prioritizer.prioritize_recommendations(
            recommendations, None
        )
        
        # Property: Both should return same number of recommendations
        assert len(prioritized_with_prefs) == len(recommendations)
        assert len(prioritized_without_prefs) == len(recommendations)
        
        # Property: All recommendations should be present in both
        with_prefs_ids = {rec.id for rec in prioritized_with_prefs}
        without_prefs_ids = {rec.id for rec in prioritized_without_prefs}
        original_ids = {rec.id for rec in recommendations}
        
        assert with_prefs_ids == original_ids
        assert without_prefs_ids == original_ids
        
        # Property: If preferences are strong, ordering might be different
        if len(recommendations) > 2:
            # Check if environmental preference affects environmental recommendations
            if user_preferences.get('prioritize_environment', False):
                env_recs = [r for r in recommendations if r.type == 'environmental']
                if env_recs:
                    # Find positions of environmental recommendations
                    env_ids = {rec.id for rec in env_recs}
                    
                    with_prefs_env_positions = [
                        i for i, rec in enumerate(prioritized_with_prefs) 
                        if rec.id in env_ids
                    ]
                    without_prefs_env_positions = [
                        i for i, rec in enumerate(prioritized_without_prefs) 
                        if rec.id in env_ids
                    ]
                    
                    if with_prefs_env_positions and without_prefs_env_positions:
                        # Environmental recs should generally be positioned better with env preference
                        avg_with_prefs = sum(with_prefs_env_positions) / len(with_prefs_env_positions)
                        avg_without_prefs = sum(without_prefs_env_positions) / len(without_prefs_env_positions)
                        
                        # Allow some flexibility due to other factors
                        assert avg_with_prefs <= avg_without_prefs + 1
    
    @given(
        easy_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={'difficulty': 'easy'})
            ),
            min_size=1, max_size=3
        ),
        complex_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={'difficulty': 'complex'})
            ),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=15, deadline=5000)
    @pytest.mark.asyncio
    async def test_difficulty_consideration(self, prioritizer, easy_recs, complex_recs):
        """
        Property: Implementation difficulty should be considered in prioritization.
        
        Given similar savings, easier recommendations should be prioritized
        higher than complex ones.
        """
        # Normalize savings to be similar
        similar_savings = EstimatedSavings(
            annual_cost_usd=500.0,
            annual_kwh=2000.0,
            co2_reduction_kg=800.0
        )
        
        normalized_easy = [
            rec.model_copy(update={'estimated_savings': similar_savings})
            for rec in easy_recs
        ]
        normalized_complex = [
            rec.model_copy(update={'estimated_savings': similar_savings})
            for rec in complex_recs
        ]
        
        all_recommendations = normalized_easy + normalized_complex
        prioritized = await prioritizer.prioritize_recommendations(all_recommendations)
        
        # Property: Easy recommendations should generally appear before complex ones
        easy_ids = {rec.id for rec in normalized_easy}
        complex_ids = {rec.id for rec in normalized_complex}
        
        easy_positions = [i for i, rec in enumerate(prioritized) if rec.id in easy_ids]
        complex_positions = [i for i, rec in enumerate(prioritized) if rec.id in complex_ids]
        
        if easy_positions and complex_positions:
            avg_easy_pos = sum(easy_positions) / len(easy_positions)
            avg_complex_pos = sum(complex_positions) / len(complex_positions)
            
            # Easy recommendations should generally be positioned better
            assert avg_easy_pos <= avg_complex_pos + 0.5  # Allow some flexibility
    
    @given(
        high_confidence_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={'confidence': 0.95})
            ),
            min_size=1, max_size=3
        ),
        low_confidence_recs=st.lists(
            optimization_recommendation_strategy().map(
                lambda x: x.model_copy(update={'confidence': 0.3})
            ),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=15, deadline=5000)
    @pytest.mark.asyncio
    async def test_confidence_influence(self, prioritizer, high_confidence_recs, low_confidence_recs):
        """
        Property: Recommendation confidence should influence prioritization.
        
        Higher confidence recommendations should generally be prioritized
        over lower confidence ones, all else being equal.
        """
        all_recommendations = high_confidence_recs + low_confidence_recs
        prioritized = await prioritizer.prioritize_recommendations(all_recommendations)
        
        # Property: High confidence recommendations should generally rank higher
        high_conf_ids = {rec.id for rec in high_confidence_recs}
        low_conf_ids = {rec.id for rec in low_confidence_recs}
        
        high_conf_positions = [i for i, rec in enumerate(prioritized) if rec.id in high_conf_ids]
        low_conf_positions = [i for i, rec in enumerate(prioritized) if rec.id in low_conf_ids]
        
        if high_conf_positions and low_conf_positions:
            avg_high_conf_pos = sum(high_conf_positions) / len(high_conf_positions)
            avg_low_conf_pos = sum(low_conf_positions) / len(low_conf_positions)
            
            # High confidence should generally be positioned better
            # Allow flexibility since other factors also matter
            assert avg_high_conf_pos <= avg_low_conf_pos + 1
    
    @given(recommendations=st.lists(optimization_recommendation_strategy(), min_size=1, max_size=10))
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.asyncio
    async def test_prioritization_stability(self, prioritizer, recommendations):
        """
        Property: Prioritization should be stable and deterministic.
        
        Running prioritization multiple times on the same input should
        produce the same ordering.
        """
        # Run prioritization twice
        prioritized1 = await prioritizer.prioritize_recommendations(recommendations.copy())
        prioritized2 = await prioritizer.prioritize_recommendations(recommendations.copy())
        
        # Property: Should produce identical results
        assert len(prioritized1) == len(prioritized2)
        
        for rec1, rec2 in zip(prioritized1, prioritized2):
            assert rec1.id == rec2.id
            assert rec1.priority == rec2.priority
    
    @given(recommendations=st.lists(optimization_recommendation_strategy(), min_size=0, max_size=2))
    @settings(max_examples=10, deadline=3000)
    @pytest.mark.asyncio
    async def test_edge_cases(self, prioritizer, recommendations):
        """
        Property: Prioritization should handle edge cases gracefully.
        
        Empty lists and single-item lists should be handled correctly.
        """
        prioritized = await prioritizer.prioritize_recommendations(recommendations)
        
        # Property: Should return same number of items
        assert len(prioritized) == len(recommendations)
        
        # Property: For empty list, should return empty list
        if len(recommendations) == 0:
            assert len(prioritized) == 0
        
        # Property: For single item, should return that item with valid priority
        if len(recommendations) == 1:
            assert len(prioritized) == 1
            assert prioritized[0].id == recommendations[0].id
            assert prioritized[0].priority in ['high', 'medium', 'low']
        
        # Property: All returned recommendations should be valid
        for rec in prioritized:
            assert isinstance(rec, OptimizationRecommendation)
            assert rec.priority in ['high', 'medium', 'low']
            assert 0.0 <= rec.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])