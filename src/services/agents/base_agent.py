"""Base agent class for the CAMEL multi-agent system."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

from ...models.energy_consumption import EnergyConsumption
from ...models.sensor_reading import SensorReading
from ...models.recommendation import OptimizationRecommendation, EstimatedSavings

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    sender_id: str
    recipient_id: str
    message_type: str  # 'data_request', 'recommendation', 'coordination', 'conflict_resolution'
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


@dataclass
class AgentContribution:
    """Tracks individual agent contributions for explainability."""
    agent_id: str
    agent_type: str
    recommendation_id: str
    contribution_type: str  # 'primary', 'supporting', 'validation'
    confidence: float
    reasoning: str
    data_sources: List[str]
    timestamp: datetime


class BaseAgent(ABC):
    """Abstract base class for all energy optimization agents."""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_queue: List[AgentMessage] = []
        self.contributions: List[AgentContribution] = []
        self.is_active = True
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
    @abstractmethod
    async def analyze_data(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Analyze energy data and generate recommendations.
        
        Args:
            consumption_data: Historical energy consumption records
            sensor_data: Real-time IoT sensor readings
            context: Additional context for analysis
            
        Returns:
            List of optimization recommendations
        """
        pass
    
    @abstractmethod
    async def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a recommendation from another agent.
        
        Args:
            recommendation: Recommendation to validate
            context: Validation context
            
        Returns:
            Validation result with confidence and feedback
        """
        pass
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message to another agent."""
        try:
            # In a real implementation, this would use a message broker
            # For now, we'll simulate message sending
            self.logger.debug(f"Sending message {message.message_id} to {message.recipient_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self, message: AgentMessage) -> bool:
        """Receive a message from another agent."""
        try:
            self.message_queue.append(message)
            self.logger.debug(f"Received message {message.message_id} from {message.sender_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return False
    
    async def process_messages(self) -> List[Dict[str, Any]]:
        """Process all pending messages."""
        results = []
        
        while self.message_queue:
            message = self.message_queue.pop(0)
            try:
                result = await self._handle_message(message)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing message {message.message_id}: {e}")
                results.append({
                    'message_id': message.message_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    async def _handle_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle a specific message based on its type."""
        if message.message_type == 'data_request':
            return await self._handle_data_request(message)
        elif message.message_type == 'recommendation':
            return await self._handle_recommendation_message(message)
        elif message.message_type == 'coordination':
            return await self._handle_coordination_message(message)
        elif message.message_type == 'conflict_resolution':
            return await self._handle_conflict_resolution(message)
        else:
            return {
                'message_id': message.message_id,
                'status': 'error',
                'error': f'Unknown message type: {message.message_type}'
            }
    
    async def _handle_data_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle data request messages."""
        return {
            'message_id': message.message_id,
            'status': 'processed',
            'response': 'Data request acknowledged'
        }
    
    async def _handle_recommendation_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle recommendation messages from other agents."""
        return {
            'message_id': message.message_id,
            'status': 'processed',
            'response': 'Recommendation received'
        }
    
    async def _handle_coordination_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle coordination messages."""
        return {
            'message_id': message.message_id,
            'status': 'processed',
            'response': 'Coordination message processed'
        }
    
    async def _handle_conflict_resolution(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle conflict resolution messages."""
        return {
            'message_id': message.message_id,
            'status': 'processed',
            'response': 'Conflict resolution processed'
        }
    
    def record_contribution(
        self,
        recommendation_id: str,
        contribution_type: str,
        confidence: float,
        reasoning: str,
        data_sources: List[str]
    ) -> None:
        """Record this agent's contribution to a recommendation."""
        contribution = AgentContribution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            recommendation_id=recommendation_id,
            contribution_type=contribution_type,
            confidence=confidence,
            reasoning=reasoning,
            data_sources=data_sources,
            timestamp=datetime.now()
        )
        self.contributions.append(contribution)
        self.logger.info(f"Recorded contribution for recommendation {recommendation_id}")
    
    def get_contributions(self, recommendation_id: Optional[str] = None) -> List[AgentContribution]:
        """Get contributions, optionally filtered by recommendation ID."""
        if recommendation_id:
            return [c for c in self.contributions if c.recommendation_id == recommendation_id]
        return self.contributions.copy()
    
    def _create_recommendation(
        self,
        rec_type: str,
        priority: str,
        title: str,
        description: str,
        implementation_steps: List[str],
        estimated_savings: EstimatedSavings,
        difficulty: str,
        confidence: float
    ) -> OptimizationRecommendation:
        """Create a new optimization recommendation."""
        rec_id = str(uuid.uuid4())
        
        recommendation = OptimizationRecommendation(
            id=rec_id,
            type=rec_type,
            priority=priority,
            title=title,
            description=description,
            implementation_steps=implementation_steps,
            estimated_savings=estimated_savings,
            difficulty=difficulty,
            agent_source=self.agent_id,
            confidence=confidence,
            created_at=datetime.now()
        )
        
        # Record this agent's contribution
        self.record_contribution(
            recommendation_id=rec_id,
            contribution_type='primary',
            confidence=confidence,
            reasoning=f"Generated by {self.agent_type} agent",
            data_sources=['energy_consumption', 'sensor_data']
        )
        
        return recommendation