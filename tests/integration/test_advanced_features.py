"""
Integration tests for advanced features including forecasting, smart home automation,
voice assistant, enterprise features, and 3D visualization.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.services.forecasting_service import EnergyForecastingService, ForecastResult
from src.services.smart_home_automation import SmartHomeAutomationService, SmartDevice, DeviceType, Protocol
from src.services.voice_assistant_integration import VoiceAssistantService, VoiceCommand, VoiceAssistant, IntentType
from src.services.enterprise_features import EnterpriseService, Tenant, User, UserRole
from src.services.visualization_3d import Visualization3DService, Scene3D, VisualizationType
from src.models.energy_consumption import EnergyConsumption
from src.models.sensor_reading import SensorReading


class TestForecastingService:
    """Test energy forecasting service."""
    
    @pytest.fixture
    def forecasting_service(self):
        """Create forecasting service instance."""
        service = EnergyForecastingService()
        return service
    
    @pytest.fixture
    def mock_energy_data(self):
        """Create mock energy consumption data."""
        data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(720):  # 30 days of hourly data
            timestamp = base_time + timedelta(hours=i)
            consumption = 45.0 + (i % 24) * 2.0 + (i % 168) * 0.5  # Daily and weekly patterns
            cost = consumption * 0.12
            
            data.append(EnergyConsumption(
                id=f"consumption_{i}",
                timestamp=timestamp,
                consumption_kwh=consumption,
                cost_usd=cost,
                source="test"
            ))
        
        return data
    
    @pytest.fixture
    def mock_sensor_data(self):
        """Create mock sensor data."""
        data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(720):
            timestamp = base_time + timedelta(hours=i)
            
            data.append(SensorReading(
                id=f"sensor_{i}",
                device_id="test_sensor",
                timestamp=timestamp,
                readings={
                    'temperature_celsius': 20.0 + (i % 24) * 0.5,
                    'humidity_percent': 50.0 + (i % 12) * 2.0,
                    'power_watts': 1000.0 + (i % 6) * 100.0,
                    'occupancy': i % 12 < 8  # Occupied 8 hours per 12-hour cycle
                }
            ))
        
        return data
    
    async def test_model_training(self, forecasting_service, mock_energy_data, mock_sensor_data):
        """Test forecasting model training."""
        # Train models
        accuracy = await forecasting_service.train_models(mock_energy_data, mock_sensor_data)
        
        # Verify training completed
        assert forecasting_service.is_trained
        assert 'consumption_mae' in accuracy
        assert 'cost_mae' in accuracy
        assert accuracy['consumption_mae'] > 0
        
        # Verify feature importance calculated
        assert 'consumption' in forecasting_service.feature_importance
        assert len(forecasting_service.feature_importance['consumption']) > 0
    
    async def test_consumption_forecasting(self, forecasting_service, mock_energy_data, mock_sensor_data):
        """Test energy consumption forecasting."""
        # Train models first
        await forecasting_service.train_models(mock_energy_data, mock_sensor_data)
        
        # Generate forecast
        forecasts = await forecasting_service.forecast_consumption(forecast_horizon_hours=24)
        
        # Verify forecast results
        assert len(forecasts) == 24
        assert all(isinstance(f, ForecastResult) for f in forecasts)
        assert all(f.predicted_consumption_kwh > 0 for f in forecasts)
        assert all(f.predicted_cost_usd > 0 for f in forecasts)
        assert all(0.0 <= f.confidence_score <= 1.0 for f in forecasts)
    
    async def test_anomaly_detection(self, forecasting_service, mock_energy_data, mock_sensor_data):
        """Test energy consumption anomaly detection."""
        # Train models
        await forecasting_service.train_models(mock_energy_data, mock_sensor_data)
        
        # Create anomalous data
        anomalous_data = mock_energy_data[-5:]
        anomalous_data[0].consumption_kwh = 200.0  # Unusually high
        anomalous_data[1].consumption_kwh = 5.0    # Unusually low
        
        # Detect anomalies
        anomalies = await forecasting_service.detect_anomalies(anomalous_data)
        
        # Verify anomaly detection
        assert len(anomalies) >= 2
        assert any(a['deviation_percent'] > 25 for a in anomalies)
        assert any(a['severity'] in ['high', 'medium'] for a in anomalies)


class TestSmartHomeAutomation:
    """Test smart home automation service."""
    
    @pytest.fixture
    def automation_service(self):
        """Create smart home automation service."""
        service = SmartHomeAutomationService()
        return service
    
    async def test_device_discovery(self, automation_service):
        """Test smart device discovery."""
        devices = await automation_service.discover_devices()
        
        # Verify devices discovered
        assert len(devices) > 0
        assert all(isinstance(d, SmartDevice) for d in devices)
        assert any(d.device_type == DeviceType.LIGHT for d in devices)
        assert any(d.protocol == Protocol.ZIGBEE for d in devices)
    
    async def test_device_control(self, automation_service):
        """Test smart device control."""
        # Discover devices first
        await automation_service.discover_devices()
        
        # Get a device to control
        device_id = list(automation_service.devices.keys())[0]
        
        # Control device
        result = await automation_service.control_device(device_id, "turn_on", {"brightness": 80})
        
        # Verify control succeeded
        assert result is True
    
    async def test_energy_optimization(self, automation_service):
        """Test energy consumption optimization."""
        # Discover devices
        await automation_service.discover_devices()
        
        # Run optimization
        results = await automation_service.optimize_energy_consumption()
        
        # Verify optimization results
        assert 'total_savings_watts' in results
        assert 'optimized_devices' in results
        assert 'recommendations' in results
        assert results['total_savings_watts'] >= 0
    
    async def test_scene_management(self, automation_service):
        """Test scene creation and activation."""
        from src.services.smart_home_automation import Scene
        
        # Discover devices first
        await automation_service.discover_devices()
        
        # Create a scene
        scene = Scene(
            scene_id="test_scene",
            name="Energy Saving Mode",
            description="Optimize for energy efficiency",
            device_states={
                list(automation_service.devices.keys())[0]: {
                    "turn_on": {},
                    "set_brightness": {"value": 50}
                }
            },
            energy_profile="eco",
            estimated_consumption_watts=0,
            is_active=False,
            created_at=datetime.now()
        )
        
        # Create and activate scene
        create_result = await automation_service.create_scene(scene)
        assert create_result is True
        
        activate_result = await automation_service.activate_scene("test_scene")
        assert activate_result is True


class TestVoiceAssistant:
    """Test voice assistant integration."""
    
    @pytest.fixture
    def voice_service(self):
        """Create voice assistant service."""
        service = VoiceAssistantService()
        return service
    
    async def test_energy_status_command(self, voice_service):
        """Test energy status voice command."""
        response = await voice_service.process_voice_command(
            "What's my current energy consumption?",
            VoiceAssistant.ALEXA,
            user_id="test_user"
        )
        
        assert response.text is not None
        assert "consumption" in response.text.lower()
        assert response.speech_text is not None
    
    async def test_device_control_command(self, voice_service):
        """Test device control voice command."""
        response = await voice_service.process_voice_command(
            "Turn on the living room light",
            VoiceAssistant.GOOGLE,
            user_id="test_user"
        )
        
        assert response.text is not None
        assert "living room light" in response.text.lower()
    
    async def test_forecast_request_command(self, voice_service):
        """Test forecast request voice command."""
        response = await voice_service.process_voice_command(
            "What will my energy usage be tomorrow?",
            VoiceAssistant.SIRI,
            user_id="test_user"
        )
        
        assert response.text is not None
        assert "tomorrow" in response.text.lower()
        assert "predict" in response.text.lower() or "forecast" in response.text.lower()
    
    async def test_recommendations_command(self, voice_service):
        """Test energy recommendations voice command."""
        response = await voice_service.process_voice_command(
            "Give me energy saving recommendations",
            VoiceAssistant.ALEXA,
            user_id="test_user"
        )
        
        assert response.text is not None
        assert "recommendation" in response.text.lower()
        assert response.card_title is not None


class TestEnterpriseFeatures:
    """Test enterprise features service."""
    
    @pytest.fixture
    def enterprise_service(self):
        """Create enterprise service."""
        service = EnterpriseService()
        return service
    
    async def test_tenant_management(self, enterprise_service):
        """Test tenant creation and management."""
        tenant_data = {
            'id': 'test_tenant_001',
            'name': 'Test Corporation',
            'domain': 'test.corp.com',
            'subscription_plan': 'enterprise',
            'max_users': 100,
            'max_devices': 1000,
            'settings': {'timezone': 'UTC', 'currency': 'USD'}
        }
        
        # Create tenant
        tenant = await enterprise_service.create_tenant(tenant_data)
        assert tenant is not None
        assert tenant.name == 'Test Corporation'
        assert tenant.max_users == 100
        
        # Retrieve tenant
        retrieved_tenant = await enterprise_service.get_tenant('test_tenant_001')
        assert retrieved_tenant is not None
        assert retrieved_tenant.id == 'test_tenant_001'
    
    async def test_user_management(self, enterprise_service):
        """Test user creation and authentication."""
        # Create tenant first
        tenant_data = {
            'id': 'test_tenant_002',
            'name': 'Test Corp 2',
            'domain': 'test2.corp.com',
            'subscription_plan': 'professional',
            'max_users': 50,
            'max_devices': 500
        }
        await enterprise_service.create_tenant(tenant_data)
        
        # Create user
        user_data = {
            'id': 'test_user_001',
            'tenant_id': 'test_tenant_002',
            'email': 'admin@test2.corp.com',
            'password': 'secure_password_123',
            'first_name': 'John',
            'last_name': 'Admin',
            'role': UserRole.TENANT_ADMIN.value
        }
        
        user = await enterprise_service.create_user(user_data)
        assert user is not None
        assert user.email == 'admin@test2.corp.com'
        assert user.role == UserRole.TENANT_ADMIN
        
        # Test authentication
        token = await enterprise_service.authenticate_user('admin@test2.corp.com', 'secure_password_123')
        assert token is not None
        
        # Verify token
        payload = await enterprise_service.verify_token(token)
        assert payload is not None
        assert payload['user_id'] == 'test_user_001'
    
    async def test_api_key_management(self, enterprise_service):
        """Test API key creation and validation."""
        key_data = {
            'key_id': 'test_key_001',
            'tenant_id': 'test_tenant_001',
            'user_id': 'test_user_001',
            'name': 'Test API Key',
            'permissions': ['read_energy_data', 'view_analytics'],
            'rate_limit': 1000
        }
        
        # Create API key
        api_key_obj = await enterprise_service.create_api_key(key_data)
        assert api_key_obj is not None
        assert api_key_obj.name == 'Test API Key'
        assert api_key_obj.rate_limit == 1000
    
    async def test_tenant_analytics(self, enterprise_service):
        """Test tenant analytics generation."""
        # Create tenant for analytics
        tenant_data = {
            'id': 'analytics_tenant',
            'name': 'Analytics Test Corp',
            'domain': 'analytics.test.com',
            'subscription_plan': 'enterprise',
            'max_users': 200,
            'max_devices': 2000
        }
        await enterprise_service.create_tenant(tenant_data)
        
        # Get analytics
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        analytics = await enterprise_service.get_tenant_analytics('analytics_tenant', start_date, end_date)
        
        assert 'tenant_id' in analytics
        assert 'users' in analytics
        assert 'api_usage' in analytics
        assert 'energy_data' in analytics
        assert analytics['tenant_id'] == 'analytics_tenant'


class Test3DVisualization:
    """Test 3D visualization service."""
    
    @pytest.fixture
    def viz_service(self):
        """Create 3D visualization service."""
        service = Visualization3DService()
        return service
    
    async def test_energy_flow_visualization(self, viz_service):
        """Test 3D energy flow visualization creation."""
        energy_data = [
            {'timestamp': datetime.now(), 'consumption': 45.2, 'device_id': 'device_001'},
            {'timestamp': datetime.now(), 'consumption': 52.1, 'device_id': 'device_002'},
            {'timestamp': datetime.now(), 'consumption': 38.7, 'device_id': 'device_003'}
        ]
        
        scene = await viz_service.create_energy_flow_visualization('building_001', energy_data)
        
        assert isinstance(scene, Scene3D)
        assert scene.name.startswith('Energy Flow')
        assert len(scene.meshes) > 0
        assert scene.metadata['visualization_type'] == VisualizationType.ENERGY_FLOW.value
    
    async def test_building_heatmap(self, viz_service):
        """Test 3D building heatmap creation."""
        consumption_data = {
            'living_room': 25.5,
            'kitchen': 18.2,
            'bedroom': 12.8,
            'bathroom': 8.1
        }
        
        scene = await viz_service.create_building_heatmap('building_001', consumption_data)
        
        assert isinstance(scene, Scene3D)
        assert scene.name.startswith('Consumption Heatmap')
        assert len(scene.meshes) >= 2  # Building + legend
        assert scene.metadata['visualization_type'] == VisualizationType.CONSUMPTION_HEATMAP.value
    
    async def test_ar_overlays(self, viz_service):
        """Test AR overlay creation."""
        device_data = [
            {'device_id': 'thermostat_001', 'type': 'thermostat', 'consumption': 15.2},
            {'device_id': 'light_001', 'type': 'light', 'consumption': 8.5},
            {'device_id': 'plug_001', 'type': 'smart_plug', 'consumption': 45.2}
        ]
        
        overlays = await viz_service.create_ar_device_overlays('building_001', device_data)
        
        assert len(overlays) == 3
        assert all(overlay.is_visible for overlay in overlays)
        assert all(overlay.device_id.startswith(('thermostat', 'light', 'plug')) for overlay in overlays)
    
    async def test_vr_dashboard(self, viz_service):
        """Test VR dashboard creation."""
        user_preferences = {
            'theme': 'dark',
            'layout': 'immersive',
            'widgets': ['energy_overview', 'device_status', 'recommendations']
        }
        
        scene = await viz_service.create_vr_dashboard(user_preferences)
        
        assert isinstance(scene, Scene3D)
        assert scene.name == 'VR Energy Dashboard'
        assert len(scene.meshes) > 0
        assert scene.metadata['render_mode'] == 'vr_headset'
    
    async def test_forecast_projection(self, viz_service):
        """Test 3D forecast projection."""
        forecast_data = []
        base_time = datetime.now()
        
        for i in range(24):
            forecast_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'predicted_consumption': 45.0 + i * 0.5,
                'confidence_lower': 40.0 + i * 0.4,
                'confidence_upper': 50.0 + i * 0.6
            })
        
        scene = await viz_service.generate_forecast_projection(forecast_data)
        
        assert isinstance(scene, Scene3D)
        assert scene.name == '3D Energy Forecast'
        assert len(scene.meshes) > 0
        assert scene.metadata['visualization_type'] == VisualizationType.FORECAST_PROJECTION.value
    
    async def test_gltf_export(self, viz_service):
        """Test glTF scene export."""
        # Create a simple scene first
        energy_data = [{'timestamp': datetime.now(), 'consumption': 45.2, 'device_id': 'test'}]
        scene = await viz_service.create_energy_flow_visualization('test_building', energy_data)
        
        # Export to glTF
        gltf_data = await viz_service.export_scene_to_gltf(scene.scene_id)
        
        assert gltf_data is not None
        assert '"asset"' in gltf_data
        assert '"version": "2.0"' in gltf_data


class TestIntegratedWorkflows:
    """Test integrated workflows across multiple advanced services."""
    
    @pytest.fixture
    def all_services(self):
        """Initialize all advanced services."""
        forecasting = EnergyForecastingService()
        smart_home = SmartHomeAutomationService()
        voice = VoiceAssistantService()
        enterprise = EnterpriseService()
        viz_3d = Visualization3DService()
        
        return {
            'forecasting': forecasting,
            'smart_home': smart_home,
            'voice': voice,
            'enterprise': enterprise,
            'viz_3d': viz_3d
        }
    
    async def test_voice_to_forecast_workflow(self, all_services):
        """Test voice command triggering forecast generation."""
        voice_service = all_services['voice']
        
        # Process voice command for forecast
        response = await voice_service.process_voice_command(
            "What will my energy usage be next week?",
            VoiceAssistant.ALEXA,
            user_id="test_user"
        )
        
        # Verify response contains forecast information
        assert response.text is not None
        assert "next week" in response.text.lower()
        assert any(word in response.text.lower() for word in ["predict", "forecast", "usage", "consumption"])
    
    async def test_smart_home_optimization_with_visualization(self, all_services):
        """Test smart home optimization with 3D visualization."""
        smart_home = all_services['smart_home']
        viz_3d = all_services['viz_3d']
        
        # Discover devices
        devices = await smart_home.discover_devices()
        assert len(devices) > 0
        
        # Run optimization
        optimization = await smart_home.optimize_energy_consumption()
        assert 'total_savings_watts' in optimization
        
        # Create visualization of optimization results
        energy_data = [
            {
                'timestamp': datetime.now(),
                'consumption': device.energy_consumption_watts,
                'device_id': device.device_id
            }
            for device in devices[:5]  # Limit for test
        ]
        
        scene = await viz_3d.create_energy_flow_visualization('test_building', energy_data)
        assert isinstance(scene, Scene3D)
    
    async def test_enterprise_multi_tenant_workflow(self, all_services):
        """Test enterprise multi-tenant workflow."""
        enterprise = all_services['enterprise']
        
        # Create multiple tenants
        tenant1_data = {
            'id': 'tenant_001',
            'name': 'Company A',
            'domain': 'companya.com',
            'subscription_plan': 'professional',
            'max_users': 50,
            'max_devices': 500
        }
        
        tenant2_data = {
            'id': 'tenant_002',
            'name': 'Company B',
            'domain': 'companyb.com',
            'subscription_plan': 'enterprise',
            'max_users': 200,
            'max_devices': 2000
        }
        
        tenant1 = await enterprise.create_tenant(tenant1_data)
        tenant2 = await enterprise.create_tenant(tenant2_data)
        
        assert tenant1 is not None
        assert tenant2 is not None
        
        # Create users for each tenant
        user1_data = {
            'id': 'user_001',
            'tenant_id': 'tenant_001',
            'email': 'admin@companya.com',
            'password': 'password123',
            'first_name': 'Alice',
            'last_name': 'Admin',
            'role': UserRole.TENANT_ADMIN.value
        }
        
        user2_data = {
            'id': 'user_002',
            'tenant_id': 'tenant_002',
            'email': 'admin@companyb.com',
            'password': 'password456',
            'first_name': 'Bob',
            'last_name': 'Admin',
            'role': UserRole.TENANT_ADMIN.value
        }
        
        user1 = await enterprise.create_user(user1_data)
        user2 = await enterprise.create_user(user2_data)
        
        assert user1 is not None
        assert user2 is not None
        assert user1.tenant_id != user2.tenant_id
        
        # Test tenant isolation
        tenant1_retrieved = await enterprise.get_tenant('tenant_001')
        tenant2_retrieved = await enterprise.get_tenant('tenant_002')
        
        assert tenant1_retrieved.id != tenant2_retrieved.id
        assert tenant1_retrieved.name != tenant2_retrieved.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])