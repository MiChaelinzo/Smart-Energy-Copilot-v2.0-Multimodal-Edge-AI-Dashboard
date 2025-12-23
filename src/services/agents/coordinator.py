"""CAMEL Multi-Agent Coordination System."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import uuid
from collections import defaultdict

from .base_agent import BaseAgent, AgentMessage, AgentContribution
from .efficiency_advisor import EfficiencyAdvisorAgent
from .cost_forecaster import CostForecasterAgent
from .eco_planner import EcoFriendlyPlannerAgent
from ...models.energy_consumption import EnergyConsumption
from ...models.sensor_reading import SensorReading
from ...models.recommendation import OptimizationRecommendation

logger = logging.getLogger(__name__)


@dataclass
class RecommendationSynthesis:
    """Synthesized recommendation with agent contributions."""
    recommendation: OptimizationRecommendation
    primary_agent: str
    supporting_agents: List[str]
    validation_scores: Dict[str, float]
    conflicts: List[str]
    synthesis_confidence: float
    created_at: datetime


@dataclass
class CollaborationSession:
    """Represents a collaborative analysis session."""
    session_id: str
    agents: List[str]
    data_sources: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    recommendations_generated: int = 0
    conflicts_resolved: int = 0
    status: str = 'active'  # 'active', 'completed', 'failed'


class CAMELAgentCoordinator:
    """
    Coordinates multiple AI agents for comprehensive energy optimization analysis.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.recommendation_history: List[RecommendationSynthesis] = []
        self.message_broker: Dict[str, List[AgentMessage]] = defaultdict(list)
        self.conflict_resolution_threshold = 0.3  # Minimum validation score difference for conflict
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            self.agents['efficiency_advisor'] = EfficiencyAdvisorAgent()
            self.agents['cost_forecaster'] = CostForecasterAgent()
            self.agents['eco_planner'] = EcoFriendlyPlannerAgent()
            
            logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    async def coordinate_analysis(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationSynthesis]:
        """
        Coordinate multi-agent analysis and synthesis.
        
        **Validates: Requirements 4.1, 4.3, 4.4**
        """
        session_id = str(uuid.uuid4())
        session = CollaborationSession(
            session_id=session_id,
            agents=list(self.agents.keys()),
            data_sources=['consumption_data'] + (['sensor_data'] if sensor_data else []),
            start_time=datetime.now()
        )
        self.active_sessions[session_id] = session
        
        try:
            logger.info(f"Starting collaborative analysis session {session_id}")
            
            # Phase 1: Individual agent analysis
            agent_recommendations = await self._collect_agent_recommendations(
                consumption_data, sensor_data, context
            )
            
            # Phase 2: Cross-agent validation
            validation_results = await self._cross_validate_recommendations(
                agent_recommendations, context or {}
            )
            
            # Phase 3: Conflict detection and resolution
            resolved_recommendations = await self._resolve_conflicts(
                agent_recommendations, validation_results
            )
            
            # Phase 4: Recommendation synthesis and prioritization
            synthesized_recommendations = await self._synthesize_recommendations(
                resolved_recommendations, validation_results
            )
            
            # Update session status
            session.end_time = datetime.now()
            session.recommendations_generated = len(synthesized_recommendations)
            session.status = 'completed'
            
            # Store in history
            self.recommendation_history.extend(synthesized_recommendations)
            
            logger.info(f"Completed collaborative analysis session {session_id} with "
                       f"{len(synthesized_recommendations)} synthesized recommendations")
            
            return synthesized_recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative analysis session {session_id}: {e}")
            session.status = 'failed'
            session.end_time = datetime.now()
            raise
    
    async def _collect_agent_recommendations(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, List[OptimizationRecommendation]]:
        """Collect recommendations from all agents in parallel."""
        agent_recommendations = {}
        
        try:
            # Run all agents in parallel
            tasks = []
            for agent_id, agent in self.agents.items():
                task = asyncio.create_task(
                    agent.analyze_data(consumption_data, sensor_data, context),
                    name=f"agent_{agent_id}"
                )
                tasks.append((agent_id, task))
            
            # Collect results
            for agent_id, task in tasks:
                try:
                    recommendations = await task
                    agent_recommendations[agent_id] = recommendations
                    logger.info(f"Agent {agent_id} generated {len(recommendations)} recommendations")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    agent_recommendations[agent_id] = []
            
            return agent_recommendations
            
        except Exception as e:
            logger.error(f"Error collecting agent recommendations: {e}")
            return {}
    
    async def _cross_validate_recommendations(
        self,
        agent_recommendations: Dict[str, List[OptimizationRecommendation]],
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Cross-validate recommendations between agents."""
        validation_results = {}
        
        try:
            # For each agent's recommendations, get validation from other agents
            for source_agent_id, recommendations in agent_recommendations.items():
                validation_results[source_agent_id] = {}
                
                for recommendation in recommendations:
                    rec_validations = {}
                    
                    # Get validation from all other agents
                    for validator_agent_id, validator_agent in self.agents.items():
                        if validator_agent_id != source_agent_id:
                            try:
                                validation = await validator_agent.validate_recommendation(
                                    recommendation, context
                                )
                                rec_validations[validator_agent_id] = validation
                            except Exception as e:
                                logger.error(f"Validation error from {validator_agent_id}: {e}")
                                rec_validations[validator_agent_id] = {
                                    'validation_score': 0.0,
                                    'feedback': [f"Validation failed: {str(e)}"],
                                    'conflicts': []
                                }
                    
                    validation_results[source_agent_id][recommendation.id] = rec_validations
            
            logger.info("Completed cross-validation of all recommendations")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    async def _resolve_conflicts(
        self,
        agent_recommendations: Dict[str, List[OptimizationRecommendation]],
        validation_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, List[OptimizationRecommendation]]:
        """
        Detect and resolve conflicts between agent recommendations.
        
        **Validates: Requirements 4.2**
        """
        resolved_recommendations = {}
        conflicts_detected = 0
        conflicts_resolved = 0
        
        try:
            for agent_id, recommendations in agent_recommendations.items():
                resolved_recommendations[agent_id] = []
                
                for recommendation in recommendations:
                    rec_id = recommendation.id
                    
                    # Check validation scores for conflicts
                    validations = validation_results.get(agent_id, {}).get(rec_id, {})
                    validation_scores = [
                        v.get('validation_score', 0.0) for v in validations.values()
                    ]
                    
                    if not validation_scores:
                        # No validation data, keep recommendation as-is
                        resolved_recommendations[agent_id].append(recommendation)
                        continue
                    
                    avg_validation_score = sum(validation_scores) / len(validation_scores)
                    min_validation_score = min(validation_scores)
                    
                    # Detect conflicts (significant disagreement between agents)
                    has_conflict = (
                        max(validation_scores) - min_validation_score > self.conflict_resolution_threshold
                        or any(v.get('conflicts', []) for v in validations.values())
                    )
                    
                    if has_conflict:
                        conflicts_detected += 1
                        logger.info(f"Conflict detected for recommendation {rec_id}")
                        
                        # Resolve conflict based on consensus and confidence
                        if avg_validation_score >= 0.6:  # Majority support
                            # Keep recommendation but adjust confidence
                            recommendation.confidence *= avg_validation_score
                            resolved_recommendations[agent_id].append(recommendation)
                            conflicts_resolved += 1
                            logger.info(f"Conflict resolved: keeping recommendation with adjusted confidence")
                        
                        elif recommendation.confidence > 0.8 and min_validation_score > 0.3:
                            # High confidence from source agent, moderate validation
                            recommendation.confidence *= 0.8  # Reduce confidence
                            resolved_recommendations[agent_id].append(recommendation)
                            conflicts_resolved += 1
                            logger.info(f"Conflict resolved: keeping high-confidence recommendation")
                        
                        else:
                            # Low consensus, remove recommendation
                            logger.info(f"Conflict unresolved: removing recommendation due to low consensus")
                    
                    else:
                        # No conflict, keep recommendation
                        resolved_recommendations[agent_id].append(recommendation)
            
            logger.info(f"Conflict resolution: {conflicts_detected} detected, {conflicts_resolved} resolved")
            
            # Update session statistics
            for session in self.active_sessions.values():
                if session.status == 'active':
                    session.conflicts_resolved = conflicts_resolved
                    break
            
            return resolved_recommendations
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return agent_recommendations  # Return original if resolution fails
    
    async def _synthesize_recommendations(
        self,
        agent_recommendations: Dict[str, List[OptimizationRecommendation]],
        validation_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> List[RecommendationSynthesis]:
        """
        Synthesize and prioritize recommendations from all agents.
        
        **Validates: Requirements 4.3**
        """
        synthesized = []
        
        try:
            # Collect all recommendations with their validation data
            all_recommendations = []
            for agent_id, recommendations in agent_recommendations.items():
                for recommendation in recommendations:
                    validations = validation_results.get(agent_id, {}).get(recommendation.id, {})
                    validation_scores = {
                        validator_id: v.get('validation_score', 0.0)
                        for validator_id, v in validations.items()
                    }
                    
                    all_recommendations.append({
                        'recommendation': recommendation,
                        'primary_agent': agent_id,
                        'validation_scores': validation_scores,
                        'validations': validations
                    })
            
            # Remove duplicates and similar recommendations
            unique_recommendations = await self._deduplicate_recommendations(all_recommendations)
            
            # Create synthesis objects
            for rec_data in unique_recommendations:
                recommendation = rec_data['recommendation']
                primary_agent = rec_data['primary_agent']
                validation_scores = rec_data['validation_scores']
                validations = rec_data['validations']
                
                # Identify supporting agents (high validation scores)
                supporting_agents = [
                    agent_id for agent_id, score in validation_scores.items()
                    if score >= 0.7
                ]
                
                # Collect conflicts
                conflicts = []
                for validator_id, validation in validations.items():
                    conflicts.extend(validation.get('conflicts', []))
                
                # Calculate synthesis confidence
                if validation_scores:
                    avg_validation = sum(validation_scores.values()) / len(validation_scores)
                    synthesis_confidence = (recommendation.confidence + avg_validation) / 2
                else:
                    synthesis_confidence = recommendation.confidence
                
                synthesis = RecommendationSynthesis(
                    recommendation=recommendation,
                    primary_agent=primary_agent,
                    supporting_agents=supporting_agents,
                    validation_scores=validation_scores,
                    conflicts=conflicts,
                    synthesis_confidence=synthesis_confidence,
                    created_at=datetime.now()
                )
                
                synthesized.append(synthesis)
            
            # Sort by priority and synthesis confidence
            synthesized.sort(
                key=lambda x: (
                    {'high': 3, 'medium': 2, 'low': 1}[x.recommendation.priority],
                    x.synthesis_confidence
                ),
                reverse=True
            )
            
            logger.info(f"Synthesized {len(synthesized)} unique recommendations")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error synthesizing recommendations: {e}")
            return []
    
    async def _deduplicate_recommendations(
        self,
        all_recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate and similar recommendations.
        
        **Validates: Requirements 4.2**
        """
        unique_recommendations = []
        seen_titles = set()
        
        try:
            for rec_data in all_recommendations:
                recommendation = rec_data['recommendation']
                title_lower = recommendation.title.lower()
                
                # Simple deduplication based on title similarity
                is_duplicate = False
                for seen_title in seen_titles:
                    # Check for significant overlap in title words
                    title_words = set(title_lower.split())
                    seen_words = set(seen_title.split())
                    
                    if title_words and seen_words:
                        overlap = len(title_words.intersection(seen_words))
                        similarity = overlap / min(len(title_words), len(seen_words))
                        
                        if similarity > 0.7:  # 70% word overlap
                            is_duplicate = True
                            logger.info(f"Duplicate recommendation detected: '{recommendation.title}'")
                            break
                
                if not is_duplicate:
                    unique_recommendations.append(rec_data)
                    seen_titles.add(title_lower)
            
            logger.info(f"Deduplication: {len(all_recommendations)} -> {len(unique_recommendations)} recommendations")
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return all_recommendations
    
    async def trigger_reanalysis(
        self,
        new_data: List[EnergyConsumption],
        trigger_reason: str = "new_data"
    ) -> List[RecommendationSynthesis]:
        """
        Trigger collaborative re-analysis when new data arrives.
        
        **Validates: Requirements 4.5**
        """
        try:
            logger.info(f"Triggering re-analysis due to: {trigger_reason}")
            
            # Check if re-analysis is needed based on data significance
            if not await self._should_reanalyze(new_data, trigger_reason):
                logger.info("Re-analysis not needed based on data significance")
                return []
            
            # Perform new analysis
            context = {
                'trigger_reason': trigger_reason,
                'previous_recommendations': len(self.recommendation_history),
                'reanalysis': True
            }
            
            new_recommendations = await self.coordinate_analysis(
                new_data, context=context
            )
            
            logger.info(f"Re-analysis completed with {len(new_recommendations)} new recommendations")
            return new_recommendations
            
        except Exception as e:
            logger.error(f"Error in re-analysis: {e}")
            return []
    
    async def _should_reanalyze(
        self,
        new_data: List[EnergyConsumption],
        trigger_reason: str
    ) -> bool:
        """Determine if re-analysis is warranted based on new data."""
        try:
            # Always re-analyze if no previous recommendations
            if not self.recommendation_history:
                return True
            
            # Check time since last analysis
            last_analysis = max(r.created_at for r in self.recommendation_history)
            time_since_last = datetime.now() - last_analysis
            
            # Re-analyze if it's been more than 24 hours
            if time_since_last > timedelta(hours=24):
                return True
            
            # Check data significance
            if len(new_data) > 10:  # Significant amount of new data
                return True
            
            # Check for anomalies in new data
            if new_data:
                consumptions = [d.consumption_kwh for d in new_data]
                avg_consumption = sum(consumptions) / len(consumptions)
                
                # Re-analyze if consumption is significantly different
                if any(abs(c - avg_consumption) > avg_consumption * 0.5 for c in consumptions):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining re-analysis need: {e}")
            return True  # Default to re-analyzing on error
    
    def get_agent_contributions(
        self,
        recommendation_id: Optional[str] = None
    ) -> List[AgentContribution]:
        """
        Get agent contributions for explainability.
        
        **Validates: Requirements 4.4**
        """
        try:
            all_contributions = []
            
            for agent in self.agents.values():
                contributions = agent.get_contributions(recommendation_id)
                all_contributions.extend(contributions)
            
            # Sort by timestamp
            all_contributions.sort(key=lambda x: x.timestamp, reverse=True)
            
            return all_contributions
            
        except Exception as e:
            logger.error(f"Error getting agent contributions: {e}")
            return []
    
    def get_collaboration_history(self) -> List[CollaborationSession]:
        """Get history of collaboration sessions."""
        try:
            sessions = list(self.active_sessions.values())
            sessions.sort(key=lambda x: x.start_time, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting collaboration history: {e}")
            return []
    
    def get_recommendation_synthesis_history(self) -> List[RecommendationSynthesis]:
        """Get history of synthesized recommendations."""
        try:
            return sorted(self.recommendation_history, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting recommendation history: {e}")
            return []


# Global coordinator instance
_coordinator = None

async def get_agent_coordinator() -> CAMELAgentCoordinator:
    """Get the global agent coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = CAMELAgentCoordinator()
    return _coordinator