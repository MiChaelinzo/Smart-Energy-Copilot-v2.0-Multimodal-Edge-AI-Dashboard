"""Cost Forecaster Agent - Analyzes pricing patterns and cost-saving opportunities."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import statistics
import calendar

from .base_agent import BaseAgent
from ...models.energy_consumption import EnergyConsumption
from ...models.sensor_reading import SensorReading
from ...models.recommendation import OptimizationRecommendation, EstimatedSavings

logger = logging.getLogger(__name__)


class CostForecasterAgent(BaseAgent):
    """Agent specialized in cost analysis and financial optimization."""
    
    def __init__(self, agent_id: str = "cost_forecaster"):
        super().__init__(agent_id, "CostForecaster")
        self.pricing_data = {
            'peak_rate': 0.18,      # $/kWh during peak hours
            'off_peak_rate': 0.09,  # $/kWh during off-peak hours
            'demand_charge': 15.0,  # $/kW for peak demand
            'peak_hours': list(range(16, 21)),  # 4 PM to 9 PM
            'seasonal_multiplier': {
                'summer': 1.2,  # June, July, August
                'winter': 1.1,  # December, January, February
                'spring_fall': 1.0  # March, April, May, September, October, November
            }
        }
    
    async def analyze_data(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Analyze energy data to identify cost-saving opportunities.
        
        **Validates: Requirements 4.1, 4.2**
        """
        recommendations = []
        
        try:
            if not consumption_data:
                self.logger.warning("No consumption data provided for cost analysis")
                return recommendations
            
            # Analyze time-of-use optimization opportunities
            tou_recommendations = await self._analyze_time_of_use(consumption_data)
            recommendations.extend(tou_recommendations)
            
            # Analyze seasonal cost patterns
            seasonal_recommendations = await self._analyze_seasonal_patterns(consumption_data)
            recommendations.extend(seasonal_recommendations)
            
            # Analyze demand charge optimization
            demand_recommendations = await self._analyze_demand_charges(consumption_data)
            recommendations.extend(demand_recommendations)
            
            # Analyze rate plan optimization
            rate_recommendations = await self._analyze_rate_plans(consumption_data)
            recommendations.extend(rate_recommendations)
            
            # Analyze cost trends and forecasts
            trend_recommendations = await self._analyze_cost_trends(consumption_data)
            recommendations.extend(trend_recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} cost optimization recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in cost analysis: {e}")
            return recommendations
    
    async def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate recommendations from other agents from a cost perspective.
        
        **Validates: Requirements 4.2**
        """
        try:
            validation_result = {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'cost_impact': 'unknown',
                'payback_period_months': None,
                'feedback': [],
                'conflicts': [],
                'timestamp': datetime.now()
            }
            
            # Validate cost savings estimates
            annual_savings = recommendation.estimated_savings.annual_cost_usd
            
            if annual_savings > 0:
                validation_result['validation_score'] = 0.8
                validation_result['cost_impact'] = 'positive'
                validation_result['feedback'].append(f"Estimated annual savings: ${annual_savings:.2f}")
                
                # Calculate payback period if implementation cost is mentioned
                if 'cost' in recommendation.description.lower() or recommendation.difficulty == 'complex':
                    # Estimate implementation cost based on difficulty
                    implementation_cost = {
                        'easy': 50,
                        'moderate': 500,
                        'complex': 2000
                    }.get(recommendation.difficulty, 500)
                    
                    if annual_savings > 0:
                        payback_months = (implementation_cost / annual_savings) * 12
                        validation_result['payback_period_months'] = payback_months
                        
                        if payback_months <= 12:
                            validation_result['feedback'].append("Excellent payback period (≤1 year)")
                        elif payback_months <= 36:
                            validation_result['feedback'].append("Good payback period (≤3 years)")
                        else:
                            validation_result['feedback'].append("Long payback period (>3 years)")
                            validation_result['validation_score'] *= 0.7
            
            elif annual_savings == 0:
                validation_result['validation_score'] = 0.5
                validation_result['cost_impact'] = 'neutral'
                validation_result['feedback'].append("No direct cost savings, but may have other benefits")
            
            else:
                validation_result['validation_score'] = 0.2
                validation_result['cost_impact'] = 'negative'
                validation_result['conflicts'].append("Recommendation may increase costs")
            
            # Check for cost-related conflicts
            if recommendation.type == 'environmental' and annual_savings < 100:
                validation_result['feedback'].append("Environmental benefit with minimal cost impact")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating recommendation: {e}")
            return {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'cost_impact': 'unknown',
                'payback_period_months': None,
                'feedback': [f"Validation error: {str(e)}"],
                'conflicts': [],
                'timestamp': datetime.now()
            }
    
    async def _analyze_time_of_use(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze time-of-use patterns to identify cost optimization opportunities."""
        recommendations = []
        
        try:
            if len(consumption_data) < 7:  # Need at least a week of data
                return recommendations
            
            # Analyze consumption by hour
            peak_consumption = 0
            off_peak_consumption = 0
            peak_cost = 0
            off_peak_cost = 0
            
            for record in consumption_data:
                hour = record.timestamp.hour
                consumption = record.consumption_kwh
                
                if hour in self.pricing_data['peak_hours']:
                    peak_consumption += consumption
                    peak_cost += consumption * self.pricing_data['peak_rate']
                else:
                    off_peak_consumption += consumption
                    off_peak_cost += consumption * self.pricing_data['off_peak_rate']
            
            total_consumption = peak_consumption + off_peak_consumption
            total_cost = peak_cost + off_peak_cost
            
            if total_consumption == 0:
                return recommendations
            
            peak_percentage = peak_consumption / total_consumption
            
            # If significant consumption during peak hours, recommend load shifting
            if peak_percentage > 0.3:  # More than 30% during peak hours
                # Calculate potential savings from shifting 50% of peak load to off-peak
                shiftable_consumption = peak_consumption * 0.5
                current_peak_cost = shiftable_consumption * self.pricing_data['peak_rate']
                new_off_peak_cost = shiftable_consumption * self.pricing_data['off_peak_rate']
                daily_savings = current_peak_cost - new_off_peak_cost
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=daily_savings * 365,
                    annual_kwh=0,  # No energy savings, just cost savings
                    co2_reduction_kg=0  # No direct CO2 reduction from load shifting
                )
                
                recommendation = self._create_recommendation(
                    rec_type='cost_saving',
                    priority='high',
                    title="Optimize Time-of-Use Energy Consumption",
                    description=f"Currently {peak_percentage:.1%} of consumption occurs during peak hours "
                              f"(${self.pricing_data['peak_rate']:.2f}/kWh vs ${self.pricing_data['off_peak_rate']:.2f}/kWh off-peak). "
                              f"Shifting flexible loads to off-peak hours could save ${daily_savings:.2f} per day.",
                    implementation_steps=[
                        "Identify flexible appliances (dishwasher, laundry, water heater)",
                        "Install programmable timers or smart switches",
                        "Schedule high-energy tasks for off-peak hours (before 4 PM or after 9 PM)",
                        "Consider time-of-use rate plan if not already enrolled",
                        "Monitor consumption patterns to verify cost savings"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.85
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing time-of-use patterns: {e}")
            return []
    
    async def _analyze_seasonal_patterns(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze seasonal cost patterns and recommend optimizations."""
        recommendations = []
        
        try:
            if len(consumption_data) < 90:  # Need at least 3 months of data
                return recommendations
            
            # Group consumption by season
            seasonal_data = {'summer': [], 'winter': [], 'spring_fall': []}
            
            for record in consumption_data:
                month = record.timestamp.month
                if month in [6, 7, 8]:  # Summer
                    seasonal_data['summer'].append(record)
                elif month in [12, 1, 2]:  # Winter
                    seasonal_data['winter'].append(record)
                else:  # Spring/Fall
                    seasonal_data['spring_fall'].append(record)
            
            # Calculate average consumption and costs by season
            seasonal_averages = {}
            for season, records in seasonal_data.items():
                if records:
                    avg_consumption = statistics.mean(r.consumption_kwh for r in records)
                    avg_cost = statistics.mean(r.cost_usd for r in records)
                    seasonal_averages[season] = {
                        'consumption': avg_consumption,
                        'cost': avg_cost,
                        'records': len(records)
                    }
            
            # Find the most expensive season
            if len(seasonal_averages) >= 2:
                max_cost_season = max(seasonal_averages.keys(), key=lambda s: seasonal_averages[s]['cost'])
                min_cost_season = min(seasonal_averages.keys(), key=lambda s: seasonal_averages[s]['cost'])
                
                max_cost = seasonal_averages[max_cost_season]['cost']
                min_cost = seasonal_averages[min_cost_season]['cost']
                
                # If significant seasonal variation, recommend seasonal optimization
                if max_cost > min_cost * 1.3:  # 30% difference
                    cost_difference = max_cost - min_cost
                    
                    estimated_savings = EstimatedSavings(
                        annual_cost_usd=cost_difference * 0.3 * 365 / 3,  # 30% reduction for 1/3 of year
                        annual_kwh=0,  # Savings depend on specific measures
                        co2_reduction_kg=0
                    )
                    
                    seasonal_tips = {
                        'summer': [
                            "Optimize air conditioning settings (78°F when home, 85°F when away)",
                            "Use ceiling fans to reduce AC load",
                            "Close blinds during peak sun hours",
                            "Schedule high-energy appliances for cooler evening hours"
                        ],
                        'winter': [
                            "Optimize heating settings (68°F when home, 60°F when away)",
                            "Seal air leaks around windows and doors",
                            "Use programmable thermostat with setback schedules",
                            "Take advantage of solar heat gain during the day"
                        ],
                        'spring_fall': [
                            "Use natural ventilation instead of HVAC when possible",
                            "Adjust thermostat settings for mild weather",
                            "Perform HVAC maintenance before peak seasons"
                        ]
                    }
                    
                    recommendation = self._create_recommendation(
                        rec_type='cost_saving',
                        priority='medium',
                        title=f"Optimize {max_cost_season.title()} Energy Costs",
                        description=f"{max_cost_season.title()} energy costs are {((max_cost/min_cost-1)*100):.0f}% "
                                  f"higher than {min_cost_season}. Implementing seasonal optimization strategies "
                                  f"could reduce costs during expensive periods.",
                        implementation_steps=seasonal_tips.get(max_cost_season, [
                            "Analyze seasonal consumption patterns",
                            "Implement season-specific energy saving measures",
                            "Monitor and adjust strategies based on results"
                        ]),
                        estimated_savings=estimated_savings,
                        difficulty='easy',
                        confidence=0.75
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {e}")
            return []
    
    async def _analyze_demand_charges(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze demand charges and recommend peak demand reduction strategies."""
        recommendations = []
        
        try:
            if len(consumption_data) < 30:  # Need at least a month of data
                return recommendations
            
            # Calculate monthly peak demand
            monthly_peaks = {}
            for record in consumption_data:
                month_key = f"{record.timestamp.year}-{record.timestamp.month:02d}"
                if month_key not in monthly_peaks:
                    monthly_peaks[month_key] = 0
                monthly_peaks[month_key] = max(monthly_peaks[month_key], record.consumption_kwh)
            
            if not monthly_peaks:
                return recommendations
            
            avg_peak_demand = statistics.mean(monthly_peaks.values())
            max_peak_demand = max(monthly_peaks.values())
            
            # Calculate current demand charges
            monthly_demand_charge = max_peak_demand * self.pricing_data['demand_charge']
            
            # If demand charges are significant, recommend demand management
            if monthly_demand_charge > 100:  # More than $100/month in demand charges
                # Potential savings from 20% peak demand reduction
                potential_reduction = max_peak_demand * 0.2
                monthly_savings = potential_reduction * self.pricing_data['demand_charge']
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=monthly_savings * 12,
                    annual_kwh=0,  # Demand management doesn't necessarily reduce total energy
                    co2_reduction_kg=0
                )
                
                recommendation = self._create_recommendation(
                    rec_type='cost_saving',
                    priority='high',
                    title="Reduce Peak Demand Charges",
                    description=f"Current peak demand of {max_peak_demand:.1f} kW results in "
                              f"${monthly_demand_charge:.2f}/month in demand charges. "
                              f"Implementing demand management could save ${monthly_savings:.2f}/month.",
                    implementation_steps=[
                        "Install demand monitoring and alerting system",
                        "Identify and prioritize high-demand equipment",
                        "Implement load shedding for non-critical devices during peak periods",
                        "Stagger startup of large equipment to avoid simultaneous operation",
                        "Consider energy storage to shave peak demand"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='complex',
                    confidence=0.8
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing demand charges: {e}")
            return []
    
    async def _analyze_rate_plans(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze current rate plan and recommend optimal rate structure."""
        recommendations = []
        
        try:
            if len(consumption_data) < 30:  # Need at least a month of data
                return recommendations
            
            # Calculate costs under different rate plans
            total_consumption = sum(r.consumption_kwh for r in consumption_data)
            current_total_cost = sum(r.cost_usd for r in consumption_data)
            
            if total_consumption == 0:
                return recommendations
            
            current_avg_rate = current_total_cost / total_consumption
            
            # Simulate time-of-use rate plan
            tou_cost = 0
            for record in consumption_data:
                hour = record.timestamp.hour
                if hour in self.pricing_data['peak_hours']:
                    tou_cost += record.consumption_kwh * self.pricing_data['peak_rate']
                else:
                    tou_cost += record.consumption_kwh * self.pricing_data['off_peak_rate']
            
            # Compare rate plans
            cost_difference = current_total_cost - tou_cost
            
            if cost_difference > 50:  # More than $50 potential savings
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=cost_difference * 12,  # Extrapolate to annual
                    annual_kwh=0,
                    co2_reduction_kg=0
                )
                
                recommendation = self._create_recommendation(
                    rec_type='cost_saving',
                    priority='medium',
                    title="Switch to Time-of-Use Rate Plan",
                    description=f"Current average rate is ${current_avg_rate:.3f}/kWh. "
                              f"Switching to time-of-use rates could save ${cost_difference:.2f} "
                              f"per month based on current usage patterns.",
                    implementation_steps=[
                        "Contact utility company to inquire about time-of-use rates",
                        "Review rate plan options and terms",
                        "Analyze potential savings based on usage patterns",
                        "Enroll in optimal rate plan",
                        "Adjust usage patterns to maximize savings"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.7
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing rate plans: {e}")
            return []
    
    async def _analyze_cost_trends(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze cost trends and provide forecasting recommendations."""
        recommendations = []
        
        try:
            if len(consumption_data) < 60:  # Need at least 2 months of data
                return recommendations
            
            # Sort data by timestamp
            sorted_data = sorted(consumption_data, key=lambda x: x.timestamp)
            
            # Calculate monthly costs
            monthly_costs = {}
            for record in sorted_data:
                month_key = f"{record.timestamp.year}-{record.timestamp.month:02d}"
                if month_key not in monthly_costs:
                    monthly_costs[month_key] = 0
                monthly_costs[month_key] += record.cost_usd
            
            if len(monthly_costs) < 2:
                return recommendations
            
            # Calculate trend
            months = sorted(monthly_costs.keys())
            costs = [monthly_costs[month] for month in months]
            
            # Simple linear trend calculation
            if len(costs) >= 3:
                recent_avg = statistics.mean(costs[-2:])  # Last 2 months
                earlier_avg = statistics.mean(costs[:-2])  # Earlier months
                
                if recent_avg > earlier_avg * 1.1:  # 10% increase
                    trend_increase = recent_avg - earlier_avg
                    
                    estimated_savings = EstimatedSavings(
                        annual_cost_usd=trend_increase * 6,  # Potential to slow trend
                        annual_kwh=0,
                        co2_reduction_kg=0
                    )
                    
                    recommendation = self._create_recommendation(
                        rec_type='cost_saving',
                        priority='medium',
                        title="Address Rising Energy Costs",
                        description=f"Energy costs have increased by ${trend_increase:.2f}/month "
                                  f"({((recent_avg/earlier_avg-1)*100):.1f}%) in recent months. "
                                  f"Implementing cost control measures could help mitigate this trend.",
                        implementation_steps=[
                            "Conduct detailed energy audit to identify cost drivers",
                            "Implement immediate low-cost efficiency measures",
                            "Monitor usage patterns more closely",
                            "Consider energy efficiency upgrades with good payback periods",
                            "Explore alternative rate plans or energy suppliers"
                        ],
                        estimated_savings=estimated_savings,
                        difficulty='moderate',
                        confidence=0.6
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing cost trends: {e}")
            return []