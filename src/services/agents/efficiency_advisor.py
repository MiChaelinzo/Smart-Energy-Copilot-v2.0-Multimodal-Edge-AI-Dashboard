"""Efficiency Advisor Agent - Focuses on energy waste reduction and device optimization."""

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


class EfficiencyAdvisorAgent(BaseAgent):
    """Agent specialized in identifying energy efficiency opportunities."""
    
    def __init__(self, agent_id: str = "efficiency_advisor"):
        super().__init__(agent_id, "EfficiencyAdvisor")
        self.efficiency_thresholds = {
            'high_consumption_threshold': 0.8,  # 80th percentile
            'idle_power_threshold': 5.0,  # watts
            'efficiency_rating_threshold': 'C',  # Below this is inefficient
            'usage_optimization_threshold': 0.3  # 30% improvement potential
        }
    
    async def analyze_data(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Analyze energy data to identify efficiency improvement opportunities.
        
        **Validates: Requirements 4.1, 4.2**
        """
        recommendations = []
        
        try:
            if not consumption_data:
                self.logger.warning("No consumption data provided for efficiency analysis")
                return recommendations
            
            # Analyze device efficiency
            device_recommendations = await self._analyze_device_efficiency(consumption_data)
            recommendations.extend(device_recommendations)
            
            # Analyze usage patterns for optimization
            pattern_recommendations = await self._analyze_usage_patterns(consumption_data, sensor_data)
            recommendations.extend(pattern_recommendations)
            
            # Analyze standby power consumption
            if sensor_data:
                standby_recommendations = await self._analyze_standby_consumption(sensor_data)
                recommendations.extend(standby_recommendations)
            
            # Analyze peak demand optimization
            peak_recommendations = await self._analyze_peak_demand(consumption_data)
            recommendations.extend(peak_recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} efficiency recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in efficiency analysis: {e}")
            return recommendations
    
    async def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate recommendations from other agents from an efficiency perspective.
        
        **Validates: Requirements 4.2**
        """
        try:
            validation_result = {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'efficiency_impact': 'unknown',
                'feedback': [],
                'conflicts': [],
                'timestamp': datetime.now()
            }
            
            # Check if recommendation aligns with efficiency principles
            if recommendation.type == 'efficiency':
                validation_result['validation_score'] = 0.9
                validation_result['efficiency_impact'] = 'high'
                validation_result['feedback'].append("Recommendation aligns with efficiency goals")
            elif recommendation.type == 'cost_saving':
                # Cost savings often align with efficiency
                validation_result['validation_score'] = 0.7
                validation_result['efficiency_impact'] = 'medium'
                validation_result['feedback'].append("Cost savings may improve efficiency")
            elif recommendation.type == 'environmental':
                # Environmental recommendations usually improve efficiency
                validation_result['validation_score'] = 0.8
                validation_result['efficiency_impact'] = 'medium'
                validation_result['feedback'].append("Environmental benefits often include efficiency gains")
            
            # Check for potential conflicts with efficiency goals
            if 'increase' in recommendation.description.lower() and 'consumption' in recommendation.description.lower():
                validation_result['conflicts'].append("Recommendation may increase energy consumption")
                validation_result['validation_score'] *= 0.5
            
            # Validate estimated savings are realistic
            if recommendation.estimated_savings.annual_kwh > 10000:  # Very high savings
                validation_result['feedback'].append("Savings estimate seems optimistic, recommend verification")
                validation_result['validation_score'] *= 0.9
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating recommendation: {e}")
            return {
                'agent_id': self.agent_id,
                'recommendation_id': recommendation.id,
                'validation_score': 0.0,
                'efficiency_impact': 'unknown',
                'feedback': [f"Validation error: {str(e)}"],
                'conflicts': [],
                'timestamp': datetime.now()
            }
    
    async def _analyze_device_efficiency(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze individual device efficiency and recommend improvements."""
        recommendations = []
        
        try:
            # Collect all device consumption data
            device_consumptions = {}
            for record in consumption_data:
                if record.device_breakdown:
                    for device in record.device_breakdown:
                        if device.device_id not in device_consumptions:
                            device_consumptions[device.device_id] = []
                        device_consumptions[device.device_id].append(device)
            
            # Analyze each device
            for device_id, device_records in device_consumptions.items():
                if len(device_records) < 2:  # Need multiple records for analysis
                    continue
                
                avg_consumption = statistics.mean(d.consumption_kwh for d in device_records)
                avg_efficiency = statistics.mean(d.usage_hours for d in device_records if d.usage_hours > 0)
                
                # Check for inefficient devices
                device_type = device_records[0].device_type
                efficiency_rating = device_records[0].efficiency_rating
                
                if efficiency_rating and efficiency_rating in ['D', 'E', 'F']:
                    # Recommend device upgrade
                    estimated_savings = EstimatedSavings(
                        annual_cost_usd=avg_consumption * 0.3 * 365 * 0.12,  # 30% savings, $0.12/kWh
                        annual_kwh=avg_consumption * 0.3 * 365,
                        co2_reduction_kg=avg_consumption * 0.3 * 365 * 0.4  # 0.4 kg CO2/kWh
                    )
                    
                    recommendation = self._create_recommendation(
                        rec_type='efficiency',
                        priority='high',
                        title=f"Upgrade Inefficient {device_type}",
                        description=f"Device {device_id} has low efficiency rating ({efficiency_rating}). "
                                  f"Upgrading to an Energy Star certified model could reduce consumption by 30%.",
                        implementation_steps=[
                            f"Research Energy Star certified {device_type} models",
                            "Compare energy consumption ratings and features",
                            "Calculate payback period for upgrade",
                            "Schedule installation of new efficient device",
                            "Properly dispose of old device"
                        ],
                        estimated_savings=estimated_savings,
                        difficulty='moderate',
                        confidence=0.85
                    )
                    recommendations.append(recommendation)
                
                # Check for devices with high standby consumption
                if avg_consumption > 0 and avg_efficiency < 8:  # Less than 8 hours active per day
                    standby_consumption = avg_consumption * (24 - avg_efficiency) / 24
                    if standby_consumption > 0.1:  # More than 0.1 kWh standby per day
                        estimated_savings = EstimatedSavings(
                            annual_cost_usd=standby_consumption * 0.5 * 365 * 0.12,  # 50% standby reduction
                            annual_kwh=standby_consumption * 0.5 * 365,
                            co2_reduction_kg=standby_consumption * 0.5 * 365 * 0.4
                        )
                        
                        recommendation = self._create_recommendation(
                            rec_type='efficiency',
                            priority='medium',
                            title=f"Reduce Standby Power for {device_type}",
                            description=f"Device {device_id} consumes significant power in standby mode. "
                                      f"Implementing smart power management could reduce standby consumption.",
                            implementation_steps=[
                                "Install smart power strips with auto-shutoff",
                                "Configure device power management settings",
                                "Set up automated schedules for device operation",
                                "Monitor standby power consumption"
                            ],
                            estimated_savings=estimated_savings,
                            difficulty='easy',
                            confidence=0.75
                        )
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing device efficiency: {e}")
            return []
    
    async def _analyze_usage_patterns(
        self,
        consumption_data: List[EnergyConsumption],
        sensor_data: Optional[List[SensorReading]]
    ) -> List[OptimizationRecommendation]:
        """Analyze usage patterns to identify optimization opportunities."""
        recommendations = []
        
        try:
            if len(consumption_data) < 7:  # Need at least a week of data
                return recommendations
            
            # Analyze consumption by hour of day
            hourly_consumption = {}
            for record in consumption_data:
                hour = record.timestamp.hour
                if hour not in hourly_consumption:
                    hourly_consumption[hour] = []
                hourly_consumption[hour].append(record.consumption_kwh)
            
            # Calculate average consumption by hour
            hourly_averages = {
                hour: statistics.mean(consumptions)
                for hour, consumptions in hourly_consumption.items()
                if consumptions
            }
            
            if not hourly_averages:
                return recommendations
            
            # Find peak consumption hours
            max_consumption = max(hourly_averages.values())
            min_consumption = min(hourly_averages.values())
            peak_hours = [hour for hour, avg in hourly_averages.items() if avg > max_consumption * 0.8]
            
            # Recommend load shifting if significant peak/off-peak difference
            if max_consumption > min_consumption * 2:  # 2x difference
                potential_savings = (max_consumption - min_consumption) * 0.3  # 30% of difference
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=potential_savings * 365 * 0.12,
                    annual_kwh=potential_savings * 365,
                    co2_reduction_kg=potential_savings * 365 * 0.4
                )
                
                recommendation = self._create_recommendation(
                    rec_type='efficiency',
                    priority='medium',
                    title="Optimize Energy Usage Timing",
                    description=f"Peak consumption occurs during hours {peak_hours}. "
                              f"Shifting non-essential loads to off-peak hours could reduce costs and improve efficiency.",
                    implementation_steps=[
                        "Identify devices that can be scheduled (dishwasher, laundry, etc.)",
                        "Install programmable timers or smart switches",
                        "Set up automated schedules for off-peak operation",
                        "Monitor consumption patterns to verify improvements"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='easy',
                    confidence=0.7
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing usage patterns: {e}")
            return []
    
    async def _analyze_standby_consumption(self, sensor_data: List[SensorReading]) -> List[OptimizationRecommendation]:
        """Analyze standby power consumption from sensor data."""
        recommendations = []
        
        try:
            # Group sensor data by device
            device_readings = {}
            for reading in sensor_data:
                if reading.sensor_id not in device_readings:
                    device_readings[reading.sensor_id] = []
                device_readings[reading.sensor_id].append(reading)
            
            # Analyze each device for standby consumption
            for device_id, readings in device_readings.items():
                power_readings = [r.readings.power_watts for r in readings if r.readings.power_watts is not None]
                
                if len(power_readings) < 10:  # Need sufficient data
                    continue
                
                # Identify potential standby periods (low but non-zero power)
                min_power = min(power_readings)
                avg_power = statistics.mean(power_readings)
                
                if min_power > self.efficiency_thresholds['idle_power_threshold'] and min_power < avg_power * 0.3:
                    # Device has significant standby consumption
                    standby_hours_per_day = 16  # Assume 16 hours standby
                    daily_standby_kwh = (min_power * standby_hours_per_day) / 1000
                    
                    estimated_savings = EstimatedSavings(
                        annual_cost_usd=daily_standby_kwh * 0.7 * 365 * 0.12,  # 70% standby reduction
                        annual_kwh=daily_standby_kwh * 0.7 * 365,
                        co2_reduction_kg=daily_standby_kwh * 0.7 * 365 * 0.4
                    )
                    
                    recommendation = self._create_recommendation(
                        rec_type='efficiency',
                        priority='medium',
                        title=f"Reduce Standby Power for {readings[0].device_type}",
                        description=f"Device {device_id} consumes {min_power:.1f}W in standby mode. "
                                  f"Installing smart power management could eliminate most standby consumption.",
                        implementation_steps=[
                            "Install smart power outlet with remote control",
                            "Configure automatic shutoff schedules",
                            "Use power strips with master/slave functionality",
                            "Enable device power management features"
                        ],
                        estimated_savings=estimated_savings,
                        difficulty='easy',
                        confidence=0.8
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing standby consumption: {e}")
            return []
    
    async def _analyze_peak_demand(self, consumption_data: List[EnergyConsumption]) -> List[OptimizationRecommendation]:
        """Analyze peak demand patterns and recommend demand management strategies."""
        recommendations = []
        
        try:
            if len(consumption_data) < 30:  # Need at least a month of data
                return recommendations
            
            # Calculate daily peak consumption
            daily_peaks = {}
            for record in consumption_data:
                date = record.timestamp.date()
                if date not in daily_peaks:
                    daily_peaks[date] = 0
                daily_peaks[date] = max(daily_peaks[date], record.consumption_kwh)
            
            if not daily_peaks:
                return recommendations
            
            avg_peak = statistics.mean(daily_peaks.values())
            max_peak = max(daily_peaks.values())
            
            # If peak demand is significantly higher than average, recommend demand management
            if max_peak > avg_peak * 1.5:  # 50% higher than average
                potential_reduction = (max_peak - avg_peak) * 0.4  # 40% of excess
                
                estimated_savings = EstimatedSavings(
                    annual_cost_usd=potential_reduction * 365 * 0.15,  # Higher rate for peak demand
                    annual_kwh=potential_reduction * 365,
                    co2_reduction_kg=potential_reduction * 365 * 0.4
                )
                
                recommendation = self._create_recommendation(
                    rec_type='efficiency',
                    priority='high',
                    title="Implement Peak Demand Management",
                    description=f"Peak demand reaches {max_peak:.1f} kWh, which is {((max_peak/avg_peak-1)*100):.0f}% "
                              f"higher than average. Implementing demand management could reduce peak charges.",
                    implementation_steps=[
                        "Install demand monitoring system",
                        "Identify high-demand appliances and equipment",
                        "Implement load shedding for non-critical devices",
                        "Set up automated demand response controls",
                        "Monitor and adjust demand management settings"
                    ],
                    estimated_savings=estimated_savings,
                    difficulty='complex',
                    confidence=0.75
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing peak demand: {e}")
            return []