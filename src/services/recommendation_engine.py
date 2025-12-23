"""
Optimization Recommendation Engine

This module implements the core recommendation generation, prioritization,
and tracking functionality for energy optimization suggestions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import uuid
import json

from ..models.energy_consumption import EnergyConsumption
from ..models.sensor_reading import SensorReading
from ..models.recommendation import OptimizationRecommendation, EstimatedSavings
from ..services.ai_service import EnergyPattern, get_ai_service
from ..services.multi_agent_service import MultiAgentEnergyService
from ..database.connection import get_db_session
from ..models.recommendation import OptimizationRecommendationORM

logger = logging.getLogger(__name__)


class RecommendationCategory(str, Enum):
    """Categories for energy optimization recommendations."""
    DEVICE_EFFICIENCY = "device_efficiency"
    USAGE_PATTERNS = "usage_patterns"
    COST_OPTIMIZATION = "cost_optimization"
    ENVIRONMENTAL = "environmental"
    MAINTENANCE = "maintenance"
    AUTOMATION = "automation"


class ImpactLevel(str, Enum):
    """Impact levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RecommendationMetrics:
    """Metrics for recommendation performance tracking."""
    total_generated: int = 0
    implemented: int = 0
    dismissed: int = 0
    pending: int = 0
    average_confidence: float = 0.0
    total_estimated_savings: float = 0.0
    actual_savings: float = 0.0
    implementation_rate: float = 0.0


@dataclass
class RecommendationContext:
    """Context information for recommendation generation."""
    user_preferences: Dict[str, Any]
    historical_patterns: List[EnergyPattern]
    current_consumption: List[EnergyConsumption]
    sensor_data: List[SensorReading]
    external_factors: Dict[str, Any]
    timestamp: datetime


class RecommendationGenerator:
    """Generates optimization recommendations based on energy data and patterns."""
    
    def __init__(self):
        self.recommendation_templates = self._load_recommendation_templates()
        self.impact_calculators = self._initialize_impact_calculators()
    
    def _load_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load recommendation templates for different scenarios."""
        return {
            "high_standby_power": {
                "category": RecommendationCategory.DEVICE_EFFICIENCY,
                "title": "Reduce Standby Power Consumption",
                "description_template": "Devices are consuming {standby_power:.1f}W in standby mode. Consider using smart power strips or unplugging devices when not in use.",
                "implementation_steps": [
                    "Identify devices with high standby power consumption",
                    "Install smart power strips with automatic shutoff",
                    "Create a schedule to turn off non-essential devices",
                    "Monitor power consumption for 2 weeks to measure impact"
                ],
                "difficulty": "easy",
                "estimated_savings_multiplier": 0.15
            },
            "peak_hour_usage": {
                "category": RecommendationCategory.COST_OPTIMIZATION,
                "title": "Shift Usage Away from Peak Hours",
                "description_template": "High energy usage detected during peak hours ({peak_hours}). Shifting {percentage:.1f}% of usage to off-peak could save ${monthly_savings:.2f}/month.",
                "implementation_steps": [
                    "Identify appliances that can be scheduled",
                    "Set timers for dishwasher, washing machine, and dryer",
                    "Use programmable thermostats for HVAC systems",
                    "Consider battery storage for peak shaving"
                ],
                "difficulty": "moderate",
                "estimated_savings_multiplier": 0.25
            },
            "inefficient_hvac": {
                "category": RecommendationCategory.DEVICE_EFFICIENCY,
                "title": "Optimize HVAC System Efficiency",
                "description_template": "HVAC system shows {efficiency_loss:.1f}% efficiency loss. Regular maintenance and smart controls could improve performance.",
                "implementation_steps": [
                    "Schedule professional HVAC maintenance",
                    "Replace air filters monthly",
                    "Install programmable or smart thermostat",
                    "Seal air leaks around windows and doors",
                    "Consider upgrading to high-efficiency system"
                ],
                "difficulty": "complex",
                "estimated_savings_multiplier": 0.30
            },
            "renewable_opportunity": {
                "category": RecommendationCategory.ENVIRONMENTAL,
                "title": "Solar Energy Installation Opportunity",
                "description_template": "Your roof receives excellent solar exposure. A {system_size:.1f}kW solar system could offset {offset_percentage:.1f}% of your energy usage.",
                "implementation_steps": [
                    "Get solar assessment from certified installer",
                    "Research local incentives and rebates",
                    "Compare financing options",
                    "Schedule installation with reputable contractor",
                    "Monitor system performance after installation"
                ],
                "difficulty": "complex",
                "estimated_savings_multiplier": 0.80
            },
            "energy_waste_pattern": {
                "category": RecommendationCategory.USAGE_PATTERNS,
                "title": "Eliminate Energy Waste Patterns",
                "description_template": "Detected consistent energy waste during {waste_periods}. Implementing automated controls could eliminate this waste.",
                "implementation_steps": [
                    "Install occupancy sensors for lighting",
                    "Use smart switches with scheduling",
                    "Set up automated HVAC controls",
                    "Create energy usage alerts and notifications"
                ],
                "difficulty": "moderate",
                "estimated_savings_multiplier": 0.20
            }
        }
    
    def _initialize_impact_calculators(self) -> Dict[str, callable]:
        """Initialize impact calculation functions for different recommendation types."""
        return {
            "cost_savings": self._calculate_cost_impact,
            "energy_savings": self._calculate_energy_impact,
            "environmental_impact": self._calculate_environmental_impact,
            "implementation_difficulty": self._calculate_implementation_difficulty
        }
    
    async def generate_recommendations(
        self,
        context: RecommendationContext
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on context.
        
        **Validates: Requirements 2.3**
        """
        try:
            recommendations = []
            
            # Analyze patterns for recommendation opportunities
            opportunities = await self._identify_optimization_opportunities(context)
            
            # Generate recommendations for each opportunity
            for opportunity in opportunities:
                recommendation = await self._create_recommendation(opportunity, context)
                if recommendation:
                    recommendations.append(recommendation)
            
            # Remove duplicates and merge similar recommendations
            recommendations = self._deduplicate_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _identify_optimization_opportunities(
        self,
        context: RecommendationContext
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from energy patterns and data."""
        opportunities = []
        
        try:
            # Analyze consumption patterns
            for pattern in context.historical_patterns:
                if pattern.pattern_type == 'daily' and len(pattern.peak_hours) > 0:
                    # Peak hour usage opportunity
                    peak_consumption = sum([
                        c.consumption_kwh for c in context.current_consumption
                        if c.timestamp.hour in pattern.peak_hours
                    ])
                    
                    if peak_consumption > pattern.average_consumption * 1.5:
                        opportunities.append({
                            'type': 'peak_hour_usage',
                            'data': {
                                'peak_hours': pattern.peak_hours,
                                'peak_consumption': peak_consumption,
                                'average_consumption': pattern.average_consumption,
                                'potential_savings': peak_consumption * 0.25
                            }
                        })
                
                elif pattern.pattern_type == 'anomaly':
                    # Energy waste pattern opportunity
                    opportunities.append({
                        'type': 'energy_waste_pattern',
                        'data': {
                            'waste_periods': f"{pattern.time_range[0].strftime('%H:%M')} - {pattern.time_range[1].strftime('%H:%M')}",
                            'waste_amount': pattern.average_consumption,
                            'cost_impact': pattern.cost_impact
                        }
                    })
            
            # Analyze sensor data for device efficiency
            if context.sensor_data:
                standby_power = self._calculate_standby_power(context.sensor_data)
                if standby_power > 50:  # More than 50W standby
                    opportunities.append({
                        'type': 'high_standby_power',
                        'data': {
                            'standby_power': standby_power,
                            'annual_cost': standby_power * 24 * 365 * 0.12 / 1000  # Assuming $0.12/kWh
                        }
                    })
                
                # Check for HVAC efficiency
                hvac_efficiency = self._analyze_hvac_efficiency(context.sensor_data)
                if hvac_efficiency < 0.8:  # Less than 80% efficiency
                    opportunities.append({
                        'type': 'inefficient_hvac',
                        'data': {
                            'efficiency_loss': (1 - hvac_efficiency) * 100,
                            'current_efficiency': hvac_efficiency
                        }
                    })
            
            # Check for renewable energy opportunities
            if self._assess_renewable_potential(context):
                opportunities.append({
                    'type': 'renewable_opportunity',
                    'data': {
                        'system_size': self._calculate_optimal_solar_size(context),
                        'offset_percentage': 75,  # Estimated offset
                        'annual_generation': 8000  # Estimated kWh/year
                    }
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return []
    
    def _calculate_standby_power(self, sensor_data: List[SensorReading]) -> float:
        """Calculate average standby power consumption from sensor data."""
        standby_readings = []
        
        for reading in sensor_data:
            if 'power_watts' in reading.readings and 'occupancy' in reading.readings:
                # Consider readings when no occupancy as standby
                if not reading.readings['occupancy'] and reading.readings['power_watts'] > 0:
                    standby_readings.append(reading.readings['power_watts'])
        
        return sum(standby_readings) / len(standby_readings) if standby_readings else 0
    
    def _analyze_hvac_efficiency(self, sensor_data: List[SensorReading]) -> float:
        """Analyze HVAC system efficiency from temperature and power data."""
        # Simplified efficiency calculation
        # In practice, this would be more sophisticated
        temp_readings = []
        power_readings = []
        
        for reading in sensor_data:
            if 'temperature_celsius' in reading.readings and 'power_watts' in reading.readings:
                temp_readings.append(reading.readings['temperature_celsius'])
                power_readings.append(reading.readings['power_watts'])
        
        if not temp_readings or not power_readings:
            return 1.0  # Assume efficient if no data
        
        # Simple efficiency metric based on temperature stability vs power usage
        temp_variance = np.var(temp_readings) if len(temp_readings) > 1 else 0
        avg_power = sum(power_readings) / len(power_readings)
        
        # Lower temperature variance with reasonable power usage indicates efficiency
        efficiency = max(0.5, 1.0 - (temp_variance / 10) - (avg_power / 10000))
        return min(1.0, efficiency)
    
    def _assess_renewable_potential(self, context: RecommendationContext) -> bool:
        """Assess if renewable energy installation is viable."""
        # Simplified assessment - in practice would use location, roof data, etc.
        total_consumption = sum(c.consumption_kwh for c in context.current_consumption)
        return total_consumption > 500  # Minimum consumption for solar viability
    
    def _calculate_optimal_solar_size(self, context: RecommendationContext) -> float:
        """Calculate optimal solar system size in kW."""
        annual_consumption = sum(c.consumption_kwh for c in context.current_consumption) * 12
        # Assume 1kW system generates ~1200 kWh/year
        return min(10.0, annual_consumption / 1200)  # Cap at 10kW for residential
    
    async def _create_recommendation(
        self,
        opportunity: Dict[str, Any],
        context: RecommendationContext
    ) -> Optional[OptimizationRecommendation]:
        """Create a recommendation from an identified opportunity."""
        try:
            template = self.recommendation_templates.get(opportunity['type'])
            if not template:
                return None
            
            # Calculate estimated savings
            estimated_savings = self._calculate_estimated_savings(opportunity, template)
            
            # Determine priority based on impact and difficulty
            priority = self._calculate_priority(estimated_savings, template['difficulty'])
            
            # Format description with opportunity data
            description = template['description_template'].format(**opportunity['data'])
            
            recommendation = OptimizationRecommendation(
                id=str(uuid.uuid4()),
                type=self._map_category_to_type(template['category']),
                priority=priority,
                title=template['title'],
                description=description,
                implementation_steps=template['implementation_steps'],
                estimated_savings=estimated_savings,
                difficulty=template['difficulty'],
                agent_source='recommendation_engine',
                confidence=0.85,  # Base confidence for generated recommendations
                created_at=datetime.now(),
                status='pending'
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            return None
    
    def _calculate_estimated_savings(
        self,
        opportunity: Dict[str, Any],
        template: Dict[str, Any]
    ) -> EstimatedSavings:
        """Calculate estimated savings for a recommendation."""
        base_savings = opportunity['data'].get('potential_savings', 100)  # Default $100/year
        multiplier = template['estimated_savings_multiplier']
        
        annual_cost_savings = base_savings * multiplier
        annual_kwh_savings = annual_cost_savings / 0.12  # Assume $0.12/kWh
        co2_reduction = annual_kwh_savings * 0.4  # Assume 0.4 kg CO2/kWh
        
        return EstimatedSavings(
            annual_cost_usd=annual_cost_savings,
            annual_kwh=annual_kwh_savings,
            co2_reduction_kg=co2_reduction
        )
    
    def _calculate_priority(self, savings: EstimatedSavings, difficulty: str) -> str:
        """Calculate recommendation priority based on savings and difficulty."""
        # High savings with easy implementation = high priority
        # Low savings with complex implementation = low priority
        
        savings_score = min(3, savings.annual_cost_usd / 100)  # $100 = 1 point
        difficulty_score = {'easy': 3, 'moderate': 2, 'complex': 1}[difficulty]
        
        total_score = savings_score + difficulty_score
        
        if total_score >= 5:
            return 'high'
        elif total_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _map_category_to_type(self, category: RecommendationCategory) -> str:
        """Map recommendation category to type."""
        mapping = {
            RecommendationCategory.DEVICE_EFFICIENCY: 'efficiency',
            RecommendationCategory.USAGE_PATTERNS: 'efficiency',
            RecommendationCategory.COST_OPTIMIZATION: 'cost_saving',
            RecommendationCategory.ENVIRONMENTAL: 'environmental',
            RecommendationCategory.MAINTENANCE: 'efficiency',
            RecommendationCategory.AUTOMATION: 'efficiency'
        }
        return mapping.get(category, 'efficiency')
    
    def _deduplicate_recommendations(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Remove duplicate recommendations and merge similar ones."""
        # Simple deduplication based on title similarity
        unique_recommendations = []
        seen_titles = set()
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(rec.title)
        
        return unique_recommendations
    
    def _calculate_cost_impact(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate cost impact of a recommendation."""
        return recommendation.estimated_savings.annual_cost_usd
    
    def _calculate_energy_impact(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate energy impact of a recommendation."""
        return recommendation.estimated_savings.annual_kwh
    
    def _calculate_environmental_impact(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate environmental impact of a recommendation."""
        return recommendation.estimated_savings.co2_reduction_kg
    
    def _calculate_implementation_difficulty(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate implementation difficulty score."""
        difficulty_scores = {'easy': 1.0, 'moderate': 2.0, 'complex': 3.0}
        return difficulty_scores.get(recommendation.difficulty, 2.0)


class RecommendationPrioritizer:
    """Prioritizes recommendations based on impact, difficulty, and user preferences."""
    
    def __init__(self):
        self.priority_weights = {
            'cost_savings': 0.4,
            'energy_savings': 0.3,
            'environmental_impact': 0.2,
            'implementation_ease': 0.1
        }
    
    async def prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Prioritize recommendations based on multiple factors.
        
        **Validates: Requirements 2.4**
        """
        try:
            if not recommendations:
                return []
            
            # Adjust weights based on user preferences
            weights = self._adjust_weights_for_preferences(user_preferences)
            
            # Calculate priority scores
            scored_recommendations = []
            for rec in recommendations:
                score = self._calculate_priority_score(rec, weights)
                scored_recommendations.append((rec, score))
            
            # Sort by score (highest first)
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Update priority field based on ranking
            prioritized = []
            total_recs = len(scored_recommendations)
            
            for i, (rec, score) in enumerate(scored_recommendations):
                # Top 30% = high, middle 40% = medium, bottom 30% = low
                if i < total_recs * 0.3:
                    priority = 'high'
                elif i < total_recs * 0.7:
                    priority = 'medium'
                else:
                    priority = 'low'
                
                # Update recommendation priority
                updated_rec = rec.model_copy(update={'priority': priority})
                prioritized.append(updated_rec)
            
            logger.info(f"Prioritized {len(prioritized)} recommendations")
            return prioritized
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {e}")
            return recommendations  # Return original list if prioritization fails
    
    def _adjust_weights_for_preferences(
        self,
        user_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Adjust priority weights based on user preferences."""
        weights = self.priority_weights.copy()
        
        if not user_preferences:
            return weights
        
        # Adjust weights based on user priorities
        if user_preferences.get('prioritize_cost', False):
            weights['cost_savings'] += 0.2
            weights['environmental_impact'] -= 0.1
            weights['energy_savings'] -= 0.1
        
        if user_preferences.get('prioritize_environment', False):
            weights['environmental_impact'] += 0.2
            weights['cost_savings'] -= 0.1
            weights['energy_savings'] -= 0.1
        
        if user_preferences.get('prefer_easy_implementation', False):
            weights['implementation_ease'] += 0.2
            weights['cost_savings'] -= 0.1
            weights['energy_savings'] -= 0.1
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_priority_score(
        self,
        recommendation: OptimizationRecommendation,
        weights: Dict[str, float]
    ) -> float:
        """Calculate priority score for a recommendation."""
        # Normalize metrics to 0-1 scale
        cost_score = min(1.0, recommendation.estimated_savings.annual_cost_usd / 1000)
        energy_score = min(1.0, recommendation.estimated_savings.annual_kwh / 5000)
        env_score = min(1.0, recommendation.estimated_savings.co2_reduction_kg / 2000)
        
        # Implementation ease (inverse of difficulty)
        ease_scores = {'easy': 1.0, 'moderate': 0.6, 'complex': 0.2}
        ease_score = ease_scores.get(recommendation.difficulty, 0.5)
        
        # Calculate weighted score
        total_score = (
            cost_score * weights['cost_savings'] +
            energy_score * weights['energy_savings'] +
            env_score * weights['environmental_impact'] +
            ease_score * weights['implementation_ease']
        )
        
        # Apply confidence multiplier
        total_score *= recommendation.confidence
        
        return total_score


class RecommendationTracker:
    """Tracks recommendation implementation and measures actual savings."""
    
    def __init__(self):
        self.tracking_data = {}
    
    async def track_recommendation_progress(
        self,
        recommendation_id: str,
        status: str,
        actual_savings: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Track progress of recommendation implementation.
        
        **Validates: Requirements 2.5, 5.4**
        """
        try:
            # Update recommendation status in database
            async with get_db_session() as session:
                rec = await session.get(OptimizationRecommendationORM, recommendation_id)
                if rec:
                    rec.status = status
                    await session.commit()
            
            # Track actual savings if provided
            if actual_savings:
                self.tracking_data[recommendation_id] = {
                    'status': status,
                    'actual_savings': actual_savings,
                    'updated_at': datetime.now()
                }
            
            logger.info(f"Updated recommendation {recommendation_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking recommendation progress: {e}")
            return False
    
    async def calculate_actual_savings(
        self,
        recommendation_id: str,
        before_data: List[EnergyConsumption],
        after_data: List[EnergyConsumption]
    ) -> Dict[str, float]:
        """Calculate actual savings from before/after consumption data."""
        try:
            before_avg = sum(c.consumption_kwh for c in before_data) / len(before_data)
            after_avg = sum(c.consumption_kwh for c in after_data) / len(after_data)
            
            kwh_savings = max(0, before_avg - after_avg)
            cost_savings = kwh_savings * 0.12  # Assume $0.12/kWh
            co2_reduction = kwh_savings * 0.4  # Assume 0.4 kg CO2/kWh
            
            return {
                'annual_kwh_savings': kwh_savings * 365,
                'annual_cost_savings': cost_savings * 365,
                'co2_reduction_kg': co2_reduction * 365
            }
            
        except Exception as e:
            logger.error(f"Error calculating actual savings: {e}")
            return {'annual_kwh_savings': 0, 'annual_cost_savings': 0, 'co2_reduction_kg': 0}
    
    async def get_recommendation_metrics(self) -> RecommendationMetrics:
        """Get overall recommendation system metrics."""
        try:
            async with get_db_session() as session:
                # Query recommendation statistics
                total_recs = await session.execute(
                    "SELECT COUNT(*) FROM optimization_recommendations"
                )
                total_generated = total_recs.scalar() or 0
                
                implemented_recs = await session.execute(
                    "SELECT COUNT(*) FROM optimization_recommendations WHERE status = 'implemented'"
                )
                implemented = implemented_recs.scalar() or 0
                
                dismissed_recs = await session.execute(
                    "SELECT COUNT(*) FROM optimization_recommendations WHERE status = 'dismissed'"
                )
                dismissed = dismissed_recs.scalar() or 0
                
                pending = total_generated - implemented - dismissed
                
                avg_confidence = await session.execute(
                    "SELECT AVG(confidence) FROM optimization_recommendations"
                )
                average_confidence = avg_confidence.scalar() or 0.0
                
                total_savings = await session.execute(
                    "SELECT SUM(annual_cost_savings_usd) FROM optimization_recommendations"
                )
                total_estimated_savings = total_savings.scalar() or 0.0
                
                implementation_rate = implemented / total_generated if total_generated > 0 else 0.0
                
                return RecommendationMetrics(
                    total_generated=total_generated,
                    implemented=implemented,
                    dismissed=dismissed,
                    pending=pending,
                    average_confidence=average_confidence,
                    total_estimated_savings=total_estimated_savings,
                    actual_savings=0.0,  # Would need to calculate from tracking data
                    implementation_rate=implementation_rate
                )
                
        except Exception as e:
            logger.error(f"Error getting recommendation metrics: {e}")
            return RecommendationMetrics()


class OptimizationRecommendationEngine:
    """Main recommendation engine coordinating all recommendation functionality."""
    
    def __init__(self):
        self.generator = RecommendationGenerator()
        self.prioritizer = RecommendationPrioritizer()
        self.tracker = RecommendationTracker()
        self.multi_agent_service = None
        self.auto_update_enabled = True
        self.last_update_time = None
    
    async def initialize(self) -> bool:
        """Initialize the recommendation engine."""
        try:
            self.multi_agent_service = MultiAgentEnergyService()
            await self.multi_agent_service.initialize()
            logger.info("Optimization recommendation engine initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            return False
    
    async def generate_comprehensive_recommendations(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        force_update: bool = False
    ) -> List[OptimizationRecommendation]:
        """
        Generate comprehensive optimization recommendations.
        
        **Validates: Requirements 2.3, 2.4, 2.5**
        """
        try:
            # Check if update is needed
            if not force_update and not self._should_update():
                return await self._get_cached_recommendations()
            
            # Get AI analysis and patterns
            ai_service = await get_ai_service()
            patterns = await ai_service.pattern_analyzer.identify_patterns(
                consumption_data, sensor_data
            )
            
            # Create recommendation context
            context = RecommendationContext(
                user_preferences=user_preferences or {},
                historical_patterns=patterns,
                current_consumption=consumption_data,
                sensor_data=sensor_data or [],
                external_factors={},
                timestamp=datetime.now()
            )
            
            # Generate base recommendations
            base_recommendations = await self.generator.generate_recommendations(context)
            
            # Get multi-agent recommendations
            agent_recommendations = []
            if self.multi_agent_service:
                agent_results = await self.multi_agent_service.generate_recommendations(
                    consumption_data, sensor_data, user_preferences
                )
                
                # Convert agent results to OptimizationRecommendation objects
                for result in agent_results:
                    if 'recommendation' in result:
                        agent_recommendations.append(result['recommendation'])
            
            # Combine all recommendations
            all_recommendations = base_recommendations + agent_recommendations
            
            # Prioritize recommendations
            prioritized_recommendations = await self.prioritizer.prioritize_recommendations(
                all_recommendations, user_preferences
            )
            
            # Save to database
            await self._save_recommendations(prioritized_recommendations)
            
            # Update timestamp
            self.last_update_time = datetime.now()
            
            logger.info(f"Generated {len(prioritized_recommendations)} comprehensive recommendations")
            return prioritized_recommendations
            
        except Exception as e:
            logger.error(f"Error generating comprehensive recommendations: {e}")
            return []
    
    def _should_update(self) -> bool:
        """Check if recommendations should be updated."""
        if not self.auto_update_enabled:
            return False
        
        if not self.last_update_time:
            return True
        
        # Update if more than 24 hours since last update
        time_since_update = datetime.now() - self.last_update_time
        return time_since_update > timedelta(hours=24)
    
    async def _get_cached_recommendations(self) -> List[OptimizationRecommendation]:
        """Get cached recommendations from database."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    "SELECT * FROM optimization_recommendations WHERE status = 'pending' ORDER BY created_at DESC LIMIT 20"
                )
                rows = result.fetchall()
                
                recommendations = []
                for row in rows:
                    rec = OptimizationRecommendation(
                        id=row.id,
                        type=row.type,
                        priority=row.priority,
                        title=row.title,
                        description=row.description,
                        implementation_steps=row.implementation_steps,
                        estimated_savings=EstimatedSavings(
                            annual_cost_usd=row.annual_cost_savings_usd,
                            annual_kwh=row.annual_kwh_savings,
                            co2_reduction_kg=row.co2_reduction_kg
                        ),
                        difficulty=row.difficulty,
                        agent_source=row.agent_source,
                        confidence=row.confidence,
                        created_at=row.created_at,
                        status=row.status
                    )
                    recommendations.append(rec)
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting cached recommendations: {e}")
            return []
    
    async def _save_recommendations(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> bool:
        """Save recommendations to database."""
        try:
            async with get_db_session() as session:
                for rec in recommendations:
                    db_rec = OptimizationRecommendationORM(
                        id=rec.id,
                        type=rec.type,
                        priority=rec.priority,
                        title=rec.title,
                        description=rec.description,
                        implementation_steps=rec.implementation_steps,
                        annual_cost_savings_usd=rec.estimated_savings.annual_cost_usd,
                        annual_kwh_savings=rec.estimated_savings.annual_kwh,
                        co2_reduction_kg=rec.estimated_savings.co2_reduction_kg,
                        difficulty=rec.difficulty,
                        agent_source=rec.agent_source,
                        confidence=rec.confidence,
                        created_at=rec.created_at,
                        status=rec.status
                    )
                    session.add(db_rec)
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            return False
    
    async def update_recommendation_status(
        self,
        recommendation_id: str,
        status: str,
        actual_savings: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update recommendation status and track progress."""
        return await self.tracker.track_recommendation_progress(
            recommendation_id, status, actual_savings
        )
    
    async def get_metrics(self) -> RecommendationMetrics:
        """Get recommendation system metrics."""
        return await self.tracker.get_recommendation_metrics()


# Global recommendation engine instance
_recommendation_engine = None

async def get_recommendation_engine() -> OptimizationRecommendationEngine:
    """Get the global recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = OptimizationRecommendationEngine()
        await _recommendation_engine.initialize()
    return _recommendation_engine