"""Multi-Agent Energy Optimization Service."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from .agents.coordinator import CAMELAgentCoordinator, get_agent_coordinator, RecommendationSynthesis
from ..models.energy_consumption import EnergyConsumption
from ..models.sensor_reading import SensorReading
from ..models.recommendation import OptimizationRecommendation

logger = logging.getLogger(__name__)


class MultiAgentEnergyService:
    """
    Main service for multi-agent energy optimization recommendations.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    def __init__(self):
        self.coordinator: Optional[CAMELAgentCoordinator] = None
        self.last_analysis_time: Optional[datetime] = None
        self.recommendation_cache: List[RecommendationSynthesis] = []
        self.auto_reanalysis_enabled = True
        self.reanalysis_threshold_hours = 24
        
    async def initialize(self) -> bool:
        """Initialize the multi-agent service."""
        try:
            self.coordinator = await get_agent_coordinator()
            logger.info("Multi-agent energy service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-agent service: {e}")
            return False
    
    async def generate_recommendations(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None,
        force_reanalysis: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive energy optimization recommendations using multiple AI agents.
        
        **Validates: Requirements 4.1, 4.3**
        """
        try:
            if not self.coordinator:
                raise RuntimeError("Multi-agent service not initialized")
            
            # Check if reanalysis is needed
            if not force_reanalysis and not self._should_analyze(consumption_data):
                logger.info("Using cached recommendations")
                return self._format_recommendations(self.recommendation_cache)
            
            logger.info("Starting multi-agent analysis")
            
            # Coordinate analysis across all agents
            synthesized_recommendations = await self.coordinator.coordinate_analysis(
                consumption_data, sensor_data, context
            )
            
            # Update cache and timestamp
            self.recommendation_cache = synthesized_recommendations
            self.last_analysis_time = datetime.now()
            
            # Format for API response
            formatted_recommendations = self._format_recommendations(synthesized_recommendations)
            
            logger.info(f"Generated {len(formatted_recommendations)} multi-agent recommendations")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error generating multi-agent recommendations: {e}")
            return []
    
    async def validate_external_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate an external recommendation using all agents.
        
        **Validates: Requirements 4.2**
        """
        try:
            if not self.coordinator:
                raise RuntimeError("Multi-agent service not initialized")
            
            validation_results = {}
            
            # Get validation from each agent
            for agent_id, agent in self.coordinator.agents.items():
                try:
                    validation = await agent.validate_recommendation(
                        recommendation, context or {}
                    )
                    validation_results[agent_id] = validation
                    
                except Exception as e:
                    logger.error(f"Validation error from agent {agent_id}: {e}")
                    validation_results[agent_id] = {
                        'validation_score': 0.0,
                        'feedback': [f"Validation failed: {str(e)}"],
                        'conflicts': []
                    }
            
            # Calculate overall validation score
            scores = [v.get('validation_score', 0.0) for v in validation_results.values()]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Collect all feedback and conflicts
            all_feedback = []
            all_conflicts = []
            for validation in validation_results.values():
                all_feedback.extend(validation.get('feedback', []))
                all_conflicts.extend(validation.get('conflicts', []))
            
            return {
                'recommendation_id': recommendation.id,
                'overall_validation_score': overall_score,
                'agent_validations': validation_results,
                'consensus_feedback': all_feedback,
                'identified_conflicts': all_conflicts,
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating external recommendation: {e}")
            return {
                'recommendation_id': recommendation.id,
                'overall_validation_score': 0.0,
                'agent_validations': {},
                'consensus_feedback': [f"Validation error: {str(e)}"],
                'identified_conflicts': [],
                'validation_timestamp': datetime.now().isoformat()
            }
    
    async def trigger_reanalysis(
        self,
        new_data: List[EnergyConsumption],
        trigger_reason: str = "manual_trigger"
    ) -> List[Dict[str, Any]]:
        """
        Trigger collaborative re-analysis with new data.
        
        **Validates: Requirements 4.5**
        """
        try:
            if not self.coordinator:
                raise RuntimeError("Multi-agent service not initialized")
            
            logger.info(f"Triggering reanalysis: {trigger_reason}")
            
            # Perform reanalysis
            new_recommendations = await self.coordinator.trigger_reanalysis(
                new_data, trigger_reason
            )
            
            # Update cache if new recommendations were generated
            if new_recommendations:
                self.recommendation_cache = new_recommendations
                self.last_analysis_time = datetime.now()
            
            formatted_recommendations = self._format_recommendations(new_recommendations)
            
            logger.info(f"Reanalysis completed with {len(formatted_recommendations)} recommendations")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error in reanalysis: {e}")
            return []
    
    async def get_agent_explanations(
        self,
        recommendation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed explanations of agent contributions for transparency.
        
        **Validates: Requirements 4.4**
        """
        try:
            if not self.coordinator:
                raise RuntimeError("Multi-agent service not initialized")
            
            # Get agent contributions
            contributions = self.coordinator.get_agent_contributions(recommendation_id)
            
            # Get collaboration history
            collaboration_history = self.coordinator.get_collaboration_history()
            
            # Format explanations
            explanations = {
                'agent_contributions': [
                    {
                        'agent_id': c.agent_id,
                        'agent_type': c.agent_type,
                        'recommendation_id': c.recommendation_id,
                        'contribution_type': c.contribution_type,
                        'confidence': c.confidence,
                        'reasoning': c.reasoning,
                        'data_sources': c.data_sources,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in contributions
                ],
                'collaboration_sessions': [
                    {
                        'session_id': s.session_id,
                        'participating_agents': s.agents,
                        'data_sources': s.data_sources,
                        'start_time': s.start_time.isoformat(),
                        'end_time': s.end_time.isoformat() if s.end_time else None,
                        'recommendations_generated': s.recommendations_generated,
                        'conflicts_resolved': s.conflicts_resolved,
                        'status': s.status
                    }
                    for s in collaboration_history[:10]  # Last 10 sessions
                ],
                'agent_specializations': {
                    'efficiency_advisor': 'Focuses on energy waste reduction and device optimization',
                    'cost_forecaster': 'Analyzes pricing patterns and cost-saving opportunities',
                    'eco_planner': 'Provides environmental impact recommendations'
                }
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error getting agent explanations: {e}")
            return {
                'agent_contributions': [],
                'collaboration_sessions': [],
                'agent_specializations': {},
                'error': str(e)
            }
    
    async def get_recommendation_history(
        self,
        limit: int = 50,
        agent_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history of synthesized recommendations."""
        try:
            if not self.coordinator:
                raise RuntimeError("Multi-agent service not initialized")
            
            history = self.coordinator.get_recommendation_synthesis_history()
            
            # Apply agent filter if specified
            if agent_filter:
                history = [h for h in history if h.primary_agent == agent_filter]
            
            # Limit results
            history = history[:limit]
            
            return self._format_recommendations(history)
            
        except Exception as e:
            logger.error(f"Error getting recommendation history: {e}")
            return []
    
    def _should_analyze(self, consumption_data: List[EnergyConsumption]) -> bool:
        """Determine if new analysis is needed."""
        try:
            # Always analyze if no previous analysis
            if not self.last_analysis_time or not self.recommendation_cache:
                return True
            
            # Check if enough time has passed
            if self.auto_reanalysis_enabled:
                time_since_last = datetime.now() - self.last_analysis_time
                if time_since_last > timedelta(hours=self.reanalysis_threshold_hours):
                    return True
            
            # Check if there's significant new data
            if consumption_data:
                latest_data_time = max(d.timestamp for d in consumption_data)
                if latest_data_time > self.last_analysis_time:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining analysis need: {e}")
            return True  # Default to analyzing on error
    
    def _format_recommendations(
        self,
        synthesized_recommendations: List[RecommendationSynthesis]
    ) -> List[Dict[str, Any]]:
        """Format synthesized recommendations for API response."""
        try:
            formatted = []
            
            for synthesis in synthesized_recommendations:
                rec = synthesis.recommendation
                
                formatted_rec = {
                    'id': rec.id,
                    'type': rec.type,
                    'priority': rec.priority,
                    'title': rec.title,
                    'description': rec.description,
                    'implementation_steps': rec.implementation_steps,
                    'estimated_savings': {
                        'annual_cost_usd': rec.estimated_savings.annual_cost_usd,
                        'annual_kwh': rec.estimated_savings.annual_kwh,
                        'co2_reduction_kg': rec.estimated_savings.co2_reduction_kg
                    },
                    'difficulty': rec.difficulty,
                    'confidence': rec.confidence,
                    'created_at': rec.created_at.isoformat(),
                    'status': rec.status,
                    
                    # Multi-agent specific fields
                    'primary_agent': synthesis.primary_agent,
                    'supporting_agents': synthesis.supporting_agents,
                    'validation_scores': synthesis.validation_scores,
                    'synthesis_confidence': synthesis.synthesis_confidence,
                    'agent_conflicts': synthesis.conflicts,
                    'synthesis_timestamp': synthesis.created_at.isoformat()
                }
                
                formatted.append(formatted_rec)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting recommendations: {e}")
            return []
    
    async def configure_reanalysis(
        self,
        enabled: bool = True,
        threshold_hours: int = 24
    ) -> bool:
        """Configure automatic reanalysis settings."""
        try:
            self.auto_reanalysis_enabled = enabled
            self.reanalysis_threshold_hours = max(1, threshold_hours)  # Minimum 1 hour
            
            logger.info(f"Reanalysis configured: enabled={enabled}, threshold={threshold_hours}h")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring reanalysis: {e}")
            return False


# Global service instance
_multi_agent_service = None

async def get_multi_agent_service() -> MultiAgentEnergyService:
    """Get the global multi-agent service instance."""
    global _multi_agent_service
    if _multi_agent_service is None:
        _multi_agent_service = MultiAgentEnergyService()
        await _multi_agent_service.initialize()
    return _multi_agent_service