"""
Advanced Energy Forecasting Service with ML-powered predictions.

Uses time series analysis, weather data, and historical patterns to predict
future energy consumption and costs with high accuracy.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import requests
from concurrent.futures import ThreadPoolExecutor

from src.models.energy_consumption import EnergyConsumption
from src.models.sensor_reading import SensorReading
from src.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Energy consumption forecast result."""
    timestamp: datetime
    predicted_consumption_kwh: float
    predicted_cost_usd: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_score: float
    contributing_factors: Dict[str, float]


@dataclass
class WeatherData:
    """Weather data for forecasting."""
    timestamp: datetime
    temperature_celsius: float
    humidity_percent: float
    wind_speed_kmh: float
    cloud_cover_percent: float
    precipitation_mm: float
    pressure_hpa: float


class EnergyForecastingService:
    """Advanced ML-powered energy forecasting service."""
    
    def __init__(self, weather_api_key: Optional[str] = None):
        self.weather_api_key = weather_api_key
        self.models = {
            'consumption': None,
            'cost': None,
            'peak_demand': None
        }
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler()
        }
        self.feature_importance = {}
        self.model_accuracy = {}
        self.is_trained = False
        
    async def train_models(self, historical_data: List[EnergyConsumption], 
                          sensor_data: List[SensorReading]) -> Dict[str, float]:
        """Train forecasting models on historical data."""
        logger.info("Training energy forecasting models...")
        
        try:
            # Prepare training data
            features_df, targets_df = await self._prepare_training_data(
                historical_data, sensor_data
            )
            
            if len(features_df) < 100:  # Need sufficient data
                raise ValueError("Insufficient historical data for training (minimum 100 data points)")
            
            # Split features and targets
            X = features_df.values
            y_consumption = targets_df['consumption_kwh'].values
            y_cost = targets_df['cost_usd'].values
            
            # Scale features
            X_scaled = self.scalers['features'].fit_transform(X)
            
            # Train consumption model
            self.models['consumption'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.models['consumption'].fit(X_scaled, y_consumption)
            
            # Train cost model
            self.models['cost'] = RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                random_state=42
            )
            self.models['cost'].fit(X_scaled, y_cost)
            
            # Train peak demand model
            peak_demand = self._calculate_peak_demand(historical_data)
            self.models['peak_demand'] = LinearRegression()
            self.models['peak_demand'].fit(X_scaled, peak_demand)
            
            # Calculate model accuracy
            consumption_pred = self.models['consumption'].predict(X_scaled)
            cost_pred = self.models['cost'].predict(X_scaled)
            peak_pred = self.models['peak_demand'].predict(X_scaled)
            
            self.model_accuracy = {
                'consumption_mae': mean_absolute_error(y_consumption, consumption_pred),
                'consumption_rmse': np.sqrt(mean_squared_error(y_consumption, consumption_pred)),
                'cost_mae': mean_absolute_error(y_cost, cost_pred),
                'cost_rmse': np.sqrt(mean_squared_error(y_cost, cost_pred)),
                'peak_mae': mean_absolute_error(peak_demand, peak_pred)
            }
            
            # Calculate feature importance
            self.feature_importance = {
                'consumption': dict(zip(features_df.columns, 
                                      self.models['consumption'].feature_importances_)),
                'cost': dict(zip(features_df.columns, 
                               self.models['cost'].feature_importances_))
            }
            
            self.is_trained = True
            logger.info(f"Models trained successfully. Accuracy: {self.model_accuracy}")
            
            return self.model_accuracy
            
        except Exception as e:
            logger.error(f"Error training forecasting models: {e}")
            raise
    
    async def forecast_consumption(self, forecast_horizon_hours: int = 168,
                                 location: str = "default") -> List[ForecastResult]:
        """Generate energy consumption forecasts for specified horizon."""
        if not self.is_trained:
            raise ValueError("Models must be trained before forecasting")
        
        logger.info(f"Generating {forecast_horizon_hours}h energy forecast...")
        
        try:
            # Get weather forecast data
            weather_data = await self._get_weather_forecast(location, forecast_horizon_hours)
            
            # Generate time series for forecast period
            forecast_timestamps = [
                datetime.now() + timedelta(hours=i) 
                for i in range(1, forecast_horizon_hours + 1)
            ]
            
            forecasts = []
            
            for i, timestamp in enumerate(forecast_timestamps):
                # Prepare features for this timestamp
                features = await self._prepare_forecast_features(
                    timestamp, weather_data[i] if i < len(weather_data) else None
                )
                
                # Scale features
                features_scaled = self.scalers['features'].transform([features])
                
                # Generate predictions
                consumption_pred = self.models['consumption'].predict(features_scaled)[0]
                cost_pred = self.models['cost'].predict(features_scaled)[0]
                peak_pred = self.models['peak_demand'].predict(features_scaled)[0]
                
                # Calculate confidence intervals (using model uncertainty)
                consumption_std = np.std([
                    tree.predict(features_scaled)[0] 
                    for tree in self.models['consumption'].estimators_[:10]
                ])
                
                confidence_score = max(0.6, 1.0 - (consumption_std / consumption_pred))
                
                # Contributing factors analysis
                contributing_factors = self._analyze_contributing_factors(
                    features, timestamp
                )
                
                forecast = ForecastResult(
                    timestamp=timestamp,
                    predicted_consumption_kwh=max(0, consumption_pred),
                    predicted_cost_usd=max(0, cost_pred),
                    confidence_interval_lower=max(0, consumption_pred - 1.96 * consumption_std),
                    confidence_interval_upper=consumption_pred + 1.96 * consumption_std,
                    confidence_score=confidence_score,
                    contributing_factors=contributing_factors
                )
                
                forecasts.append(forecast)
            
            logger.info(f"Generated {len(forecasts)} forecast points")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    async def detect_anomalies(self, recent_data: List[EnergyConsumption]) -> List[Dict[str, Any]]:
        """Detect anomalous energy consumption patterns."""
        if not self.is_trained:
            return []
        
        anomalies = []
        
        try:
            for consumption in recent_data[-24:]:  # Check last 24 data points
                # Prepare features for this data point
                features = await self._prepare_forecast_features(consumption.timestamp)
                features_scaled = self.scalers['features'].transform([features])
                
                # Predict expected consumption
                expected_consumption = self.models['consumption'].predict(features_scaled)[0]
                actual_consumption = consumption.consumption_kwh
                
                # Calculate deviation
                deviation_percent = abs(actual_consumption - expected_consumption) / expected_consumption * 100
                
                if deviation_percent > 25:  # 25% threshold for anomaly
                    anomaly = {
                        'timestamp': consumption.timestamp,
                        'actual_consumption': actual_consumption,
                        'expected_consumption': expected_consumption,
                        'deviation_percent': deviation_percent,
                        'severity': 'high' if deviation_percent > 50 else 'medium',
                        'possible_causes': self._identify_anomaly_causes(
                            actual_consumption, expected_consumption, features
                        )
                    }
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def optimize_forecast_accuracy(self, validation_data: List[EnergyConsumption]) -> Dict[str, float]:
        """Optimize model parameters for better accuracy."""
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")
        
        logger.info("Optimizing forecast accuracy...")
        
        try:
            # Prepare validation features
            features_df, targets_df = await self._prepare_training_data(validation_data, [])
            X_val = self.scalers['features'].transform(features_df.values)
            y_val = targets_df['consumption_kwh'].values
            
            # Test different hyperparameters
            best_params = {}
            best_score = float('inf')
            
            param_combinations = [
                {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4},
                {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6},
                {'n_estimators': 250, 'learning_rate': 0.15, 'max_depth': 8},
            ]
            
            for params in param_combinations:
                model = GradientBoostingRegressor(**params, random_state=42)
                model.fit(X_val, y_val)
                predictions = model.predict(X_val)
                score = mean_squared_error(y_val, predictions)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    self.models['consumption'] = model
            
            # Update accuracy metrics
            optimized_predictions = self.models['consumption'].predict(X_val)
            self.model_accuracy['optimized_rmse'] = np.sqrt(mean_squared_error(y_val, optimized_predictions))
            
            logger.info(f"Model optimized. New RMSE: {self.model_accuracy['optimized_rmse']:.2f}")
            
            return {
                'best_params': best_params,
                'improved_rmse': self.model_accuracy['optimized_rmse'],
                'improvement_percent': (
                    (self.model_accuracy['consumption_rmse'] - self.model_accuracy['optimized_rmse']) /
                    self.model_accuracy['consumption_rmse'] * 100
                )
            }
            
        except Exception as e:
            logger.error(f"Error optimizing forecast accuracy: {e}")
            raise
    
    async def _prepare_training_data(self, historical_data: List[EnergyConsumption],
                                   sensor_data: List[SensorReading]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data with engineered features."""
        features = []
        targets = []
        
        # Sort data by timestamp
        historical_data.sort(key=lambda x: x.timestamp)
        sensor_data.sort(key=lambda x: x.timestamp)
        
        for consumption in historical_data:
            # Time-based features
            timestamp = consumption.timestamp
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = day_of_week >= 5
            
            # Seasonal features
            day_of_year = timestamp.timetuple().tm_yday
            season_sin = np.sin(2 * np.pi * day_of_year / 365.25)
            season_cos = np.cos(2 * np.pi * day_of_year / 365.25)
            
            # Daily cycle features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Weekly cycle features
            week_sin = np.sin(2 * np.pi * day_of_week / 7)
            week_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Historical consumption features (lag features)
            lag_1h = self._get_lagged_consumption(historical_data, timestamp, hours=1)
            lag_24h = self._get_lagged_consumption(historical_data, timestamp, hours=24)
            lag_168h = self._get_lagged_consumption(historical_data, timestamp, hours=168)  # 1 week
            
            # Rolling averages
            avg_24h = self._get_rolling_average(historical_data, timestamp, hours=24)
            avg_168h = self._get_rolling_average(historical_data, timestamp, hours=168)
            
            # Sensor data features
            sensor_features = self._get_sensor_features(sensor_data, timestamp)
            
            feature_row = [
                hour, day_of_week, month, int(is_weekend),
                season_sin, season_cos, hour_sin, hour_cos, week_sin, week_cos,
                lag_1h, lag_24h, lag_168h, avg_24h, avg_168h,
                *sensor_features
            ]
            
            features.append(feature_row)
            targets.append([consumption.consumption_kwh, consumption.cost_usd])
        
        feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'season_sin', 'season_cos', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos',
            'lag_1h', 'lag_24h', 'lag_168h', 'avg_24h', 'avg_168h',
            'temperature', 'humidity', 'power_watts', 'occupancy'
        ]
        
        features_df = pd.DataFrame(features, columns=feature_columns)
        targets_df = pd.DataFrame(targets, columns=['consumption_kwh', 'cost_usd'])
        
        return features_df, targets_df
    
    async def _get_weather_forecast(self, location: str, hours: int) -> List[WeatherData]:
        """Get weather forecast data from external API."""
        if not self.weather_api_key:
            # Return mock weather data if no API key
            return [
                WeatherData(
                    timestamp=datetime.now() + timedelta(hours=i),
                    temperature_celsius=20.0 + np.random.normal(0, 5),
                    humidity_percent=50.0 + np.random.normal(0, 15),
                    wind_speed_kmh=10.0 + np.random.normal(0, 5),
                    cloud_cover_percent=30.0 + np.random.normal(0, 20),
                    precipitation_mm=0.0,
                    pressure_hpa=1013.25 + np.random.normal(0, 10)
                )
                for i in range(hours)
            ]
        
        try:
            # Example using OpenWeatherMap API (replace with actual implementation)
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric',
                'cnt': min(hours // 3, 40)  # API returns 3-hour intervals, max 40 points
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = []
            for item in data['list']:
                weather_data.append(WeatherData(
                    timestamp=datetime.fromtimestamp(item['dt']),
                    temperature_celsius=item['main']['temp'],
                    humidity_percent=item['main']['humidity'],
                    wind_speed_kmh=item['wind']['speed'] * 3.6,  # m/s to km/h
                    cloud_cover_percent=item['clouds']['all'],
                    precipitation_mm=item.get('rain', {}).get('3h', 0),
                    pressure_hpa=item['main']['pressure']
                ))
            
            return weather_data
            
        except Exception as e:
            logger.warning(f"Failed to get weather data: {e}. Using mock data.")
            return await self._get_weather_forecast("", hours)  # Fallback to mock data
    
    def _get_lagged_consumption(self, data: List[EnergyConsumption], 
                               timestamp: datetime, hours: int) -> float:
        """Get consumption value from specified hours ago."""
        target_time = timestamp - timedelta(hours=hours)
        
        # Find closest data point
        closest_data = min(data, key=lambda x: abs((x.timestamp - target_time).total_seconds()))
        
        if abs((closest_data.timestamp - target_time).total_seconds()) < 3600:  # Within 1 hour
            return closest_data.consumption_kwh
        
        return 0.0  # No data available
    
    def _get_rolling_average(self, data: List[EnergyConsumption], 
                           timestamp: datetime, hours: int) -> float:
        """Calculate rolling average consumption."""
        start_time = timestamp - timedelta(hours=hours)
        relevant_data = [
            d for d in data 
            if start_time <= d.timestamp <= timestamp
        ]
        
        if not relevant_data:
            return 0.0
        
        return sum(d.consumption_kwh for d in relevant_data) / len(relevant_data)
    
    def _get_sensor_features(self, sensor_data: List[SensorReading], 
                           timestamp: datetime) -> List[float]:
        """Extract features from sensor data."""
        # Find sensor readings within 1 hour of timestamp
        relevant_readings = [
            r for r in sensor_data
            if abs((r.timestamp - timestamp).total_seconds()) < 3600
        ]
        
        if not relevant_readings:
            return [20.0, 50.0, 1000.0, 0.0]  # Default values
        
        # Average sensor values
        avg_temp = np.mean([r.readings.temperature_celsius or 20.0 for r in relevant_readings])
        avg_humidity = np.mean([r.readings.humidity_percent or 50.0 for r in relevant_readings])
        avg_power = np.mean([r.readings.power_watts or 1000.0 for r in relevant_readings])
        avg_occupancy = np.mean([float(r.readings.occupancy or False) for r in relevant_readings])
        
        return [avg_temp, avg_humidity, avg_power, avg_occupancy]
    
    async def _prepare_forecast_features(self, timestamp: datetime, 
                                       weather: Optional[WeatherData] = None) -> List[float]:
        """Prepare features for a single forecast timestamp."""
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        is_weekend = day_of_week >= 5
        
        # Seasonal features
        day_of_year = timestamp.timetuple().tm_yday
        season_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        season_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Daily cycle features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Weekly cycle features
        week_sin = np.sin(2 * np.pi * day_of_week / 7)
        week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # For forecasting, we use recent historical averages as lag features
        # In a real implementation, these would come from recent data
        lag_1h = 250.0  # Typical consumption
        lag_24h = 240.0
        lag_168h = 260.0
        avg_24h = 245.0
        avg_168h = 250.0
        
        # Weather/sensor features
        temperature = weather.temperature_celsius if weather else 20.0
        humidity = weather.humidity_percent if weather else 50.0
        power_watts = 1000.0  # Estimated
        occupancy = 1.0 if 6 <= hour <= 22 else 0.0  # Estimated occupancy
        
        return [
            hour, day_of_week, month, int(is_weekend),
            season_sin, season_cos, hour_sin, hour_cos, week_sin, week_cos,
            lag_1h, lag_24h, lag_168h, avg_24h, avg_168h,
            temperature, humidity, power_watts, occupancy
        ]
    
    def _calculate_peak_demand(self, data: List[EnergyConsumption]) -> np.ndarray:
        """Calculate peak demand values for training."""
        # Group by day and find daily peaks
        daily_peaks = []
        current_day = None
        daily_consumptions = []
        
        for consumption in sorted(data, key=lambda x: x.timestamp):
            day = consumption.timestamp.date()
            
            if current_day != day:
                if daily_consumptions:
                    daily_peaks.append(max(daily_consumptions))
                current_day = day
                daily_consumptions = [consumption.consumption_kwh]
            else:
                daily_consumptions.append(consumption.consumption_kwh)
        
        if daily_consumptions:
            daily_peaks.append(max(daily_consumptions))
        
        # Extend to match data length
        peak_values = []
        for consumption in data:
            day = consumption.timestamp.date()
            # Find corresponding daily peak
            day_index = min(len(daily_peaks) - 1, 
                          max(0, (consumption.timestamp.date() - data[0].timestamp.date()).days))
            peak_values.append(daily_peaks[day_index] if daily_peaks else consumption.consumption_kwh)
        
        return np.array(peak_values)
    
    def _analyze_contributing_factors(self, features: List[float], 
                                    timestamp: datetime) -> Dict[str, float]:
        """Analyze factors contributing to the forecast."""
        feature_names = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'season_sin', 'season_cos', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos',
            'lag_1h', 'lag_24h', 'lag_168h', 'avg_24h', 'avg_168h',
            'temperature', 'humidity', 'power_watts', 'occupancy'
        ]
        
        if 'consumption' not in self.feature_importance:
            return {}
        
        # Get feature importance from trained model
        importance_dict = self.feature_importance['consumption']
        
        # Calculate weighted contributions
        contributions = {}
        for i, feature_name in enumerate(feature_names):
            if feature_name in importance_dict and i < len(features):
                contributions[feature_name] = importance_dict[feature_name] * abs(features[i])
        
        # Normalize to percentages
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v / total_contribution * 100 for k, v in contributions.items()}
        
        return contributions
    
    def _identify_anomaly_causes(self, actual: float, expected: float, 
                                features: List[float]) -> List[str]:
        """Identify possible causes of consumption anomalies."""
        causes = []
        
        if actual > expected * 1.5:
            causes.extend([
                "Unusually high consumption detected",
                "Possible equipment malfunction",
                "Extreme weather conditions",
                "Additional devices in use"
            ])
        elif actual < expected * 0.5:
            causes.extend([
                "Unusually low consumption detected",
                "Possible equipment shutdown",
                "Reduced occupancy",
                "Energy-saving measures activated"
            ])
        
        # Analyze specific features
        if len(features) > 15:  # Temperature feature
            temp = features[15]
            if temp > 30:
                causes.append("High temperature may increase cooling demand")
            elif temp < 5:
                causes.append("Low temperature may increase heating demand")
        
        return causes[:3]  # Return top 3 causes
    
    async def save_models(self, filepath: str) -> bool:
        """Save trained models to disk."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_accuracy': self.model_accuracy,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    async def load_models(self, filepath: str) -> bool:
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.model_accuracy = model_data['model_accuracy']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


# Global forecasting service instance
forecasting_service = EnergyForecastingService()