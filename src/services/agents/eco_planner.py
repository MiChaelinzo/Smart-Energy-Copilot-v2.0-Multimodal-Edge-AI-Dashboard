"""Eco-Friendly Planner Agent - Provides environmental impact recommendations."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import statistics

from .base_agent import BaseAgent
from ...models.energy_consumption import EnergyConsumption
from ...models.sensor_reading import SensorReading
from ...models.recommendation import OptimizationRecommendation, EstimatedSavings

logger = logging.getLogger(__name__)


class EcoFriendlyPlannerAgent(BaseAgent):
    """Agent specialized in environmental impact analysis and sustainability recommendations."""
    
    def __init__(self, agent_id: str = "eco_planner"):
        super().__init__(agent_id, "EcoFriendlyPlanner")
        self.environmental_factors = {
            'co2_per_kwh': 0.4,  # kg CO2 per kWh (grid average)
            'renewable_co2_per_kwh': 0.05,  # kg CO2 per kWh (renewable sources)
            'peak_grid_intensity': 0.6,  # kg CO2 per kWh during peak hours
            'off_peak_grid_intensity': 0.3,  # kg CO2 per kWh during off-peak hours
            'water_per_kwh': 2.5,  # liters of water per kWh (thermoelectric cooling)
            'renewable_targets': {
                'solar_potential': 0.8,  # 80% of consumption could be solar
                'efficiency_potential': 0.3  # 30% efficiency improvement potential
            }
        }
    
    async def analyze_data(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Analyze energy data to identify environmental optimization opportunities.
        
        **Validates: Requirements 4.1, 4.2**
        """
        recommendations = []
        
        try:
            if not consumption_data:
                self.logger.warning("No consumption data provided for environmental analysis")
                return recommendations
            
            # Analyze carbon footprint reduction opportunities
            carbon_recommendations = await self._analyze_carbon_footprint(consumption_data)
            recommendations.extend(carbon_recommendations)
            
            # Analyze renewable energy opportunities
            renewable_recommendations = await self._analyze_renewable_opportunities(consumption_data)
            recommendations.extend(renewable_recommendations)
            
            # Analyze grid impact optimization
            grid_recommendations = await self._analyze_grid_impact(consumption_data)
            recommendations.extend(grid_recommendations)
            
            # Analyze resource conservation opportunities
            conservation_recommendations = await self._analyze_resource_conservation(consumption_data, sensor_data)
            recommendations.extend(conservation_recommendations)
            
            # Analyze sustainable practices
            sustainability_recommendations = await self._analyze_sustainability_practices(consumption_data)
            recommendations.extend(sustainability_recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} environmental recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in environmental analysis: {e}")
            return recommendations
    
    async def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate recommendations from other agents from an environmental perspective.
        
        **Validates: Requirements 4.2**
        """
        try:
            validation_result = {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'environmental_impact': 'unknown',
                'co2_impact_kg_annual': 0.0,
                'feedback': [],
                'conflicts': [],
                'timestamp': datetime.now()
            }
            
            # Validate environmental impact
            co2_reduction = recommendation.estimated_savings.co2_reduction_kg
            
            if co2_reduction > 0:
                validation_result['validation_score'] = 0.9
                validation_result['environmental_impact'] = 'positive'
                validation_result['co2_impact_kg_annual'] = co2_reduction
                validation_result['feedback'].append(f"Reduces CO2 emissions by {co2_reduction:.1f} kg/year")
                
                # Categorize impact level
                if co2_reduction > 1000:  # More than 1 ton CO2/year
                    validation_result['feedback'].append("Significant environmental benefit (>1 ton CO2/year)")
                elif co2_reduction > 100:
                    validation_result['feedback'].append("Moderate environmental benefit")
                else:
                    validation_result['feedback'].append("Small but positive environmental benefit")
            
            elif co2_reduction == 0:
                # Check if recommendation has indirect environmental benefits
                if any(keyword in recommendation.description.lower() 
                       for keyword in ['efficiency', 'renewable', 'sustainable', 'conservation']):
                    validation_result['validation_score'] = 0.6
                    validation_result['environmental_impact'] = 'indirect_positive'
                    validation_result['feedback'].append("May have indirect environmental benefits")
                else:
                    validation_result['validation_score'] = 0.4
                    validation_result['environmental_impact'] = 'neutral'
                    validation_result['feedback'].append("No direct environmental impact")
            
            else:
                validation_result['validation_score'] = 0.1
                validation_result['environmental_impact'] = 'negative'
                validation_result['conflicts'].append("Recommendation may increase environmental impact")
            
            # Check for potential environmental conflicts
            if any(keyword in recommendation.description.lower() 
                   for keyword in ['increase', 'more', 'additional']) and recommendation.type != 'environmental':
                validation_result['feedback'].append("Monitor for potential increased resource consumption")
            
            # Validate against sustainability principles
            if recommendation.type == 'cost_saving' and co2_reduction > 0:
                validation_result['feedback'].append("Excellent: cost savings with environmental benefits")
                validation_result['validation_score'] = min(validation_result['validation_score'] * 1.1, 1.0)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating recommendation: {e}")
            return {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'environmental_impact': 'unknown',
                'co2_impact_kg_annual': 0.0,
                'feedback': [f"Validation error: {str(e)}"],
                'conflicts': [],
                'timestamp': datetime.now()
            }
    
    async def _analyze_carbon_footprint(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze carbon footprint and recommend reduction strategies."""
        recommendations = []
        
        try:
            if not consumption_data:
                return recommendations
            
            # Calculate current carbon footprint
            total_consumption = sum(r.consumption_kwh for r in consumption_data)
            total_co2 = total_consumption * self.environmental_factors['co2_per_kwh']
            
            # Calculate daily average
            days = len(set(r.timestamp.date() for r in consumption_data))
            if days == 0:
                return recommendations
            
            daily_co2 = total_co2 / days
            annual_co2 = daily_co2 * 365
            
            # If significant carbon footprint, recommend reduction strategies
            if annual_co2 > 1000:  # More than 1 ton CO2/year
                # Recommend efficiency improvements (30% reduction potential)
                efficiency_reduction = annual_co2 * 0.3
                efficiency_kwh_savings = (total_consumption / days) * 365 * 0.3
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=efficiency_kwh_savings * 0.12,  # $0.12/kWh
                    annual_kwh=efficiency_kwh_savings,
                    co2_reduction_kg=efficiency_reduction
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='high',
                    title="Reduce Carbon Footprint Through Efficiency",
                    description=f"Current energy use generates {annual_co2:.0f} kg CO2/year. "
                              f"Implementing efficiency measures could reduce emissions by {efficiency_reduction:.0f} kg CO2/year "
                              f"(equivalent to planting {efficiency_reduction/22:.0f} trees).",
                    implementation_steps=[
                        "Conduct comprehensive energy audit",
                        "Upgrade to LED lighting throughout facility",
                        "Improve insulation and air sealing",
                        "Upgrade to ENERGY STAR certified appliances",
                        "Implement smart thermostats and controls",
                        "Monitor and track carbon footprint reduction"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='moderate',
                    confidence=0.8
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing carbon footprint: {e}")
            return []
    
    async def _analyze_renewable_opportunities(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze opportunities for renewable energy adoption."""
        recommendations = []
        
        try:
            if len(consumption_data) < 30:  # Need at least a month of data
                return recommendations
            
            # Calculate average daily consumption
            total_consumption = sum(r.consumption_kwh for r in consumption_data)
            days = len(set(r.timestamp.date() for r in consumption_data))
            daily_avg_consumption = total_consumption / days if days > 0 else 0
            
            if daily_avg_consumption == 0:
                return recommendations
            
            # Estimate solar potential (assuming 80% of consumption could be offset)
            solar_potential_kwh = daily_avg_consumption * 365 * self.environmental_factors['renewable_targets']['solar_potential']
            
            # Calculate environmental benefits
            grid_co2_avoided = solar_potential_kwh * self.environmental_factors['co2_per_kwh']
            cost_savings = solar_potential_kwh * 0.12  # $0.12/kWh
            
            # Recommend solar if consumption is significant
            if daily_avg_consumption > 10:  # More than 10 kWh/day
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=cost_savings,
                    annual_kwh=0,  # Solar generates rather than saves
                    co2_reduction_kg=grid_co2_avoided
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='medium',
                    title="Install Solar Energy System",
                    description=f"Solar panels could offset {self.environmental_factors['renewable_targets']['solar_potential']*100:.0f}% "
                              f"of energy consumption ({solar_potential_kwh:.0f} kWh/year), "
                              f"avoiding {grid_co2_avoided:.0f} kg CO2/year and saving ${cost_savings:.0f}/year.",
                    implementation_steps=[
                        "Conduct solar site assessment and shading analysis",
                        "Get quotes from certified solar installers",
                        "Evaluate financing options and incentives",
                        "Obtain necessary permits and approvals",
                        "Install solar panels and monitoring system",
                        "Connect to grid with net metering agreement"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='complex',
                    confidence=0.7
                )
                recommendations.append(recommendation)
            
            # Recommend green energy program if solar isn't feasible
            if daily_avg_consumption > 5:  # More than 5 kWh/day
                green_program_co2_reduction = total_consumption * (
                    self.environmental_factors['co2_per_kwh'] - 
                    self.environmental_factors['renewable_co2_per_kwh']
                ) * (365 / days)
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=0,  # May have small premium
                    annual_kwh=0,
                    co2_reduction_kg=green_program_co2_reduction
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='low',
                    title="Switch to Green Energy Program",
                    description=f"Enrolling in utility's renewable energy program could reduce CO2 emissions "
                              f"by {green_program_co2_reduction:.0f} kg/year with minimal cost impact.",
                    implementation_steps=[
                        "Contact utility about green energy programs",
                        "Compare renewable energy certificate options",
                        "Enroll in program with highest renewable content",
                        "Monitor environmental impact through utility reports"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.9
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing renewable opportunities: {e}")
            return []
    
    async def _analyze_grid_impact(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze grid impact and recommend load optimization for environmental benefit."""
        recommendations = []
        
        try:
            if len(consumption_data) < 7:  # Need at least a week of data
                return recommendations
            
            # Analyze consumption by time of day
            peak_hours = list(range(16, 21))  # 4 PM to 9 PM (high grid intensity)
            off_peak_hours = list(range(0, 6)) + list(range(22, 24))  # Night hours (low grid intensity)
            
            peak_consumption = 0
            off_peak_consumption = 0
            
            for record in consumption_data:
                hour = record.timestamp.hour
                if hour in peak_hours:
                    peak_consumption += record.consumption_kwh
                elif hour in off_peak_hours:
                    off_peak_consumption += record.consumption_kwh
            
            total_consumption = sum(r.consumption_kwh for r in consumption_data)
            
            if total_consumption == 0:
                return recommendations
            
            peak_percentage = peak_consumption / total_consumption
            
            # If significant peak consumption, recommend load shifting for environmental benefit
            if peak_percentage > 0.25:  # More than 25% during peak hours
                # Calculate CO2 reduction from shifting 50% of peak load to off-peak
                shiftable_consumption = peak_consumption * 0.5
                current_peak_co2 = shiftable_consumption * self.environmental_factors['peak_grid_intensity']
                new_off_peak_co2 = shiftable_consumption * self.environmental_factors['off_peak_grid_intensity']
                co2_reduction = (current_peak_co2 - new_off_peak_co2) * (365 / len(consumption_data))
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=0,  # Focus on environmental benefit
                    annual_kwh=0,
                    co2_reduction_kg=co2_reduction
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='medium',
                    title="Reduce Grid Impact Through Load Shifting",
                    description=f"Shifting energy use from peak hours (when grid is dirtier) to off-peak hours "
                              f"could reduce CO2 emissions by {co2_reduction:.0f} kg/year without reducing total consumption.",
                    implementation_steps=[
                        "Identify flexible loads that can be shifted to off-peak hours",
                        "Install timers or smart controls for water heaters, pool pumps, etc.",
                        "Schedule dishwasher, laundry, and other appliances for late evening",
                        "Use battery storage to shift solar generation to evening peak",
                        "Monitor grid carbon intensity and adjust usage accordingly"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.75
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing grid impact: {e}")
            return []
    
    async def _analyze_resource_conservation(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]]
    ) -> List[OptimizationRecommendation]:
        """Analyze resource conservation opportunities beyond just energy."""
        recommendations = []
        
        try:
            if not consumption_data:
                return recommendations
            
            # Calculate water usage associated with energy consumption
            total_consumption = sum(r.consumption_kwh for r in consumption_data)
            days = len(set(r.timestamp.date() for r in consumption_data))
            daily_consumption = total_consumption / days if days > 0 else 0
            
            # Water usage from thermoelectric power generation
            annual_water_usage = daily_consumption * 365 * self.environmental_factors['water_per_kwh']
            
            if annual_water_usage > 1000:  # More than 1000 liters/year
                # Recommend efficiency measures that also conserve water
                water_savings = annual_water_usage * 0.3  # 30% reduction potential
                energy_savings = daily_consumption * 365 * 0.3
                co2_savings = energy_savings * self.environmental_factors['co2_per_kwh']
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=energy_savings * 0.12,
                    annual_kwh=energy_savings,
                    co2_reduction_kg=co2_savings
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='medium',
                    title="Conserve Water Through Energy Efficiency",
                    description=f"Energy consumption indirectly uses {annual_water_usage:.0f} liters of water/year "
                              f"for power generation. Reducing energy use by 30% would save {water_savings:.0f} liters/year "
                              f"and {co2_savings:.0f} kg CO2/year.",
                    implementation_steps=[
                        "Focus on efficiency measures that reduce overall energy consumption",
                        "Prioritize renewable energy to reduce water-intensive grid power",
                        "Implement water-efficient appliances (dishwashers, washing machines)",
                        "Consider heat pump water heaters for dual efficiency benefits",
                        "Monitor both energy and water conservation progress"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='moderate',
                    confidence=0.7
                )
                recommendations.append(recommendation)
            
            # Analyze temperature data for HVAC optimization
            if sensor_data:
                temp_readings = [r for r in sensor_data if r.readings.temperature_celsius is not None]
                if temp_readings:
                    temperatures = [r.readings.temperature_celsius for r in temp_readings]
                    avg_temp = statistics.mean(temperatures)
                    
                    # If indoor temperature suggests over-conditioning
                    if avg_temp < 20 or avg_temp > 26:  # Too cold in winter or too hot in summer
                        hvac_savings_kwh = daily_consumption * 365 * 0.15  # 15% HVAC savings
                        hvac_co2_savings = hvac_savings_kwh * self.environmental_factors['co2_per_kwh']
                        
                        estimated_savings = EstimatedSavings(
                            annual_cost_usd=hvac_savings_kwh * 0.12,
                            annual_kwh=hvac_savings_kwh,
                            co2_reduction_kg=hvac_co2_savings
                        )
                        
                        temp_advice = "warmer" if avg_temp < 20 else "cooler"
                        
                        recommendation = self._create_recommendation(
                            rec_type='environmental',
                            priority='medium',
                            title="Optimize HVAC Settings for Sustainability",
                            description=f"Indoor temperature averaging {avg_temp:.1f}°C suggests potential for "
                                      f"more sustainable HVAC operation. Adjusting settings {temp_advice} could "
                                      f"reduce CO2 emissions by {hvac_co2_savings:.0f} kg/year.",
                            implementation_steps=[
                                "Adjust thermostat to more sustainable settings (20-22°C winter, 24-26°C summer)",
                                "Use programmable schedules to reduce conditioning when unoccupied",
                                "Improve building envelope to reduce HVAC load",
                                "Consider natural ventilation when weather permits",
                                "Monitor comfort levels and adjust gradually"
                            ],
                            estimated_savings=estimated_savings,
                            difficulty='easy',
                            confidence=0.8
                        )
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing resource conservation: {e}")
            return []
    
    async def _analyze_sustainability_practices(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze and recommend sustainable energy practices."""
        recommendations = []
        
        try:
            if len(consumption_data) < 14:  # Need at least 2 weeks of data
                return recommendations
            
            # Analyze consumption variability (consistent usage is more sustainable)
            daily_consumptions = {}
            for record in consumption_data:
                date = record.timestamp.date()
                if date not in daily_consumptions:
                    daily_consumptions[date] = 0
                daily_consumptions[date] += record.consumption_kwh
            
            if len(daily_consumptions) < 7:
                return recommendations
            
            consumptions = list(daily_consumptions.values())
            avg_consumption = statistics.mean(consumptions)
            consumption_std = statistics.stdev(consumptions) if len(consumptions) > 1 else 0
            
            # High variability suggests opportunities for better energy management
            if consumption_std > avg_consumption * 0.3:  # More than 30% variation
                # Recommend energy management practices
                management_savings_kwh = avg_consumption * 365 * 0.1  # 10% through better management
                management_co2_savings = management_savings_kwh * self.environmental_factors['co2_per_kwh']
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=management_savings_kwh * 0.12,
                    annual_kwh=management_savings_kwh,
                    co2_reduction_kg=management_co2_savings
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='low',
                    title="Implement Sustainable Energy Management Practices",
                    description=f"Energy consumption varies significantly (±{(consumption_std/avg_consumption)*100:.0f}%), "
                              f"indicating opportunities for more consistent, sustainable energy use patterns.",
                    implementation_steps=[
                        "Establish daily energy use routines and schedules",
                        "Implement energy monitoring and feedback systems",
                        "Set up automated controls for consistent operation",
                        "Educate occupants about sustainable energy practices",
                        "Track and celebrate energy conservation achievements",
                        "Participate in utility demand response programs"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.6
                )
                recommendations.append(recommendation)
            
            # Recommend environmental certification or green building practices
            if avg_consumption > 20:  # Significant energy user
                certification_co2_savings = avg_consumption * 365 * 0.2 * self.environmental_factors['co2_per_kwh']
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=0,  # Focus on environmental benefit
                    annual_kwh=0,
                    co2_reduction_kg=certification_co2_savings
                )
                
                recommendation = self._create_recommendation(
                    rec_type='environmental',
                    priority='low',
                    title="Pursue Green Building Certification",
                    description=f"Consider pursuing LEED, ENERGY STAR, or similar green building certification "
                              f"to formalize sustainability commitments and achieve systematic improvements.",
                    implementation_steps=[
                        "Research applicable green building certification programs",
                        "Conduct gap analysis against certification requirements",
                        "Develop sustainability improvement plan",
                        "Implement required efficiency and sustainability measures",
                        "Document and verify improvements for certification",
                        "Maintain ongoing sustainability practices"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='complex',
                    confidence=0.5
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing sustainability practices: {e}")
            return []