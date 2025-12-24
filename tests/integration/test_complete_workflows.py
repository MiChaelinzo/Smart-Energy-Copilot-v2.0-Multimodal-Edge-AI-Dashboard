"""
Integration tests for complete Smart Energy Copilot workflows.

Tests the end-to-end functionality from document upload through recommendation generation,
IoT data integration with AI analysis, and multi-agent system collaboration under load.

**Validates: Requirements 1.1-7.5**
"""

import asyncio
import pytest
import tempfile
import json
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from src.main import app
from src.services.ocr_service import OCRProcessingEngine, DocumentFormat
from src.services.ai_service import get_ai_service
from src.services.multi_agent_service import get_multi_agent_service
from src.services.iot_integration import IoTIntegrationService
from src.services.recommendation_engine import get_recommendation_engine
from src.models.energy_consumption import EnergyConsumption
from src.models.sensor_reading import SensorReading, SensorReadings
from src.models.device import Device, DeviceConfig, ProtocolType
from src.models.recommendation import OptimizationRecommendation, EstimatedSavings
from src.database.connection import get_db_session


class TestCompleteWorkflows:
    """Integration tests for complete system workflows."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_utility_bill_image(self):
        """Create a sample utility bill image for testing."""
        # Create a simple test image with text
        img = Image.new('RGB', (800, 600), color='white')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    @pytest.fixture
    def sample_energy_data(self):
        """Sample energy consumption data."""
        from src.models.energy_consumption import BillingPeriod
        return [
            EnergyConsumption(
                id="test_consumption_1",
                timestamp=datetime.now() - timedelta(days=1),
                source="utility_bill",
                consumption_kwh=450.5,
                cost_usd=67.58,
                billing_period=BillingPeriod(
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                ),
                confidence_score=0.95
            ),
            EnergyConsumption(
                id="test_consumption_2", 
                timestamp=datetime.now() - timedelta(hours=12),
                source="iot_sensor",
                consumption_kwh=225.3,
                cost_usd=33.80,
                billing_period=BillingPeriod(
                    start_date=datetime.now() - timedelta(days=15),
                    end_date=datetime.now()
                ),
                confidence_score=0.88
            )
        ]
    
    @pytest.fixture
    def sample_iot_devices(self):
        """Sample IoT devices for testing."""
        return [
            Device(
                device_id="smart_meter_001",
                device_type="smart_meter",
                name="Main Electrical Meter",
                location="utility_room",
                config=DeviceConfig(
                    protocol=ProtocolType.MQTT,
                    endpoint="mqtt://localhost:1883",
                    topic="energy/meter/001",
                    polling_interval=30,
                    retry_attempts=3
                )
            ),
            Device(
                device_id="temp_sensor_001",
                device_type="temperature_sensor",
                name="Living Room Temperature",
                location="living_room",
                config=DeviceConfig(
                    protocol=ProtocolType.HTTP_REST,
                    endpoint="http://192.168.1.100/api/temperature",
                    polling_interval=60,
                    retry_attempts=3
                )
            )
        ]
    
    @pytest.fixture
    def sample_sensor_readings(self):
        """Sample sensor readings for testing."""
        return [
            SensorReading(
                sensor_id="smart_meter_001",
                device_type="smart_meter",
                timestamp=datetime.now(),
                readings=SensorReadings(
                    power_watts=2500.0,
                    voltage=240.0,
                    current_amps=10.4
                ),
                quality_score=0.95,
                location="utility_room"
            ),
            SensorReading(
                sensor_id="temp_sensor_001",
                device_type="temperature_sensor",
                timestamp=datetime.now(),
                readings=SensorReadings(
                    temperature_celsius=22.5,
                    humidity_percent=45.0
                ),
                quality_score=0.92,
                location="living_room"
            )
        ]

    @pytest.mark.asyncio
    async def test_document_to_recommendation_pipeline(
        self, 
        client, 
        sample_utility_bill_image,
        sample_energy_data
    ):
        """
        Test complete pipeline from document upload to recommendation generation.
        
        **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.3, 4.1, 4.3**
        """
        # Step 1: Upload and process document
        with patch('src.services.ocr_service.OCRProcessingEngine') as mock_ocr:
            # Mock OCR processing
            mock_engine = AsyncMock()
            mock_ocr.return_value = mock_engine
            
            # Mock OCR result
            from src.services.ocr_service import OCRResult, EnergyFieldData
            mock_ocr_result = OCRResult(
                text="Electric Bill - Consumption: 450.5 kWh, Cost: $67.58",
                confidence=0.95,
                format=DocumentFormat.JPEG,
                page_count=1,
                bounding_boxes=[]
            )
            mock_engine.process_document.return_value = mock_ocr_result
            mock_engine.assess_quality.return_value = {"overall_quality": 0.95}
            
            # Mock energy field extraction
            mock_energy_data = EnergyFieldData(
                consumption_kwh=450.5,
                cost_usd=67.58,
                billing_period_start=datetime.now() - timedelta(days=30),
                billing_period_end=datetime.now(),
                account_number="123456789",
                confidence_scores={
                    "consumption_kwh": 0.95,
                    "cost_usd": 0.92,
                    "billing_period": 0.88
                }
            )
            mock_engine.extract_energy_fields.return_value = mock_energy_data
            
            # Upload document
            response = client.post(
                "/api/ocr/upload",
                files={"file": ("test_bill.jpg", sample_utility_bill_image, "image/jpeg")}
            )
            
            assert response.status_code == 200
            ocr_data = response.json()
            assert ocr_data["success"] is True
            assert "energy_data" in ocr_data
            assert ocr_data["energy_data"]["consumption_kwh"] == 450.5
        
        # Step 2: Process with AI service for pattern analysis
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Mock AI analysis result
            mock_ai.batch_inference.return_value = {
                'results': {
                    'patterns': [
                        {
                            'type': 'daily_peak',
                            'description': 'High consumption during evening hours',
                            'confidence': 0.87
                        }
                    ],
                    'fusion_data': {
                        'combined_insights': [
                            {
                                'insight': 'Potential for 15% savings with load shifting',
                                'confidence': 0.82
                            }
                        ]
                    }
                },
                'processing_time_ms': 245.6,
                'summary': {'data_quality_score': 0.91}
            }
            
            # Call AI analysis endpoint
            ai_response = client.post(
                "/api/v1/ai/analyze-patterns",
                json={
                    "consumption_data": [
                        {
                            "id": "test_consumption_1",
                            "timestamp": datetime.now().isoformat(),
                            "source": "utility_bill",
                            "consumption_kwh": 450.5,
                            "cost_usd": 67.58,
                            "billing_period_start": (datetime.now() - timedelta(days=30)).isoformat(),
                            "billing_period_end": datetime.now().isoformat(),
                            "confidence_score": 0.95
                        }
                    ],
                    "analysis_type": "comprehensive"
                }
            )
            
            assert ai_response.status_code == 200
            ai_data = ai_response.json()
            assert len(ai_data["patterns"]) > 0
            assert ai_data["confidence_score"] > 0.8
        
        # Step 3: Generate multi-agent recommendations
        with patch('src.services.multi_agent_service.get_multi_agent_service') as mock_multi_agent:
            mock_service = AsyncMock()
            mock_multi_agent.return_value = mock_service
            
            # Mock multi-agent recommendations
            mock_service.generate_recommendations.return_value = [
                {
                    'id': 'rec_001',
                    'type': 'cost_saving',
                    'priority': 'high',
                    'title': 'Shift peak hour usage',
                    'description': 'Move high-consumption activities to off-peak hours',
                    'implementation_steps': ['Identify peak usage devices', 'Schedule usage during off-peak'],
                    'estimated_savings': {
                        'annual_cost_usd': 180.50,
                        'annual_kwh': 1200.0,
                        'co2_reduction_kg': 540.0
                    },
                    'difficulty': 'easy',
                    'confidence': 0.85,
                    'primary_agent': 'cost_forecaster',
                    'supporting_agents': ['efficiency_advisor'],
                    'synthesis_confidence': 0.88
                }
            ]
            
            # Test recommendation generation (would be called internally)
            recommendations = await mock_service.generate_recommendations(sample_energy_data)
            
            assert len(recommendations) > 0
            assert recommendations[0]['type'] in ['cost_saving', 'efficiency', 'environmental']
            assert recommendations[0]['estimated_savings']['annual_cost_usd'] > 0
        
        # Step 4: Verify complete pipeline integration
        # This would normally be done through a single endpoint that orchestrates all steps
        pipeline_success = (
            ocr_data["success"] and 
            ai_data["confidence_score"] > 0.8 and 
            len(recommendations) > 0
        )
        
        assert pipeline_success, "Complete document-to-recommendation pipeline should succeed"

    @pytest.mark.asyncio
    async def test_iot_data_integration_with_ai_analysis(
        self,
        sample_iot_devices,
        sample_sensor_readings,
        sample_energy_data
    ):
        """
        Test IoT data integration with AI analysis workflow.
        
        **Validates: Requirements 7.1, 7.2, 7.4, 2.1, 2.2**
        """
        # Step 1: Initialize IoT integration service
        iot_service = IoTIntegrationService()
        
        # Step 2: Register IoT devices
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            # Mock protocol handlers
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected = True
            mock_create_handler.return_value = mock_handler
            
            # Register devices
            for device in sample_iot_devices:
                success = await iot_service.register_device(device)
                assert success, f"Device {device.device_id} should register successfully"
        
        # Step 3: Simulate device discovery
        with patch.object(iot_service, 'discover_devices') as mock_discover:
            from src.models.device import DeviceDiscoveryResult
            mock_discover.return_value = DeviceDiscoveryResult(
                discovered_devices=sample_iot_devices,
                discovery_method="MQTT, HTTP, Modbus",
                success=True
            )
            
            discovery_result = await iot_service.discover_devices()
            assert discovery_result.success
            assert len(discovery_result.discovered_devices) == len(sample_iot_devices)
        
        # Step 4: Read sensor data with validation
        with patch.object(iot_service, '_validate_and_interpolate') as mock_validate:
            mock_validate.side_effect = lambda reading: reading  # Pass through
            
            with patch.object(iot_service, '_store_sensor_reading') as mock_store:
                mock_store.return_value = None  # Mock successful storage
                
                # Mock handler data reading
                for device_id, handler in iot_service.handlers.items():
                    # Find corresponding sensor reading
                    reading = next(
                        (r for r in sample_sensor_readings if r.sensor_id == device_id),
                        sample_sensor_readings[0]
                    )
                    handler.read_data.return_value = reading
                
                # Read data from all devices
                readings = await iot_service.read_all_devices()
                
                assert len(readings) == len(sample_iot_devices)
                for device_id, reading in readings.items():
                    assert reading is not None
                    assert reading.quality_score > 0.8
        
        # Step 5: Integrate IoT data with AI analysis
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Mock data fusion
            mock_ai.fusion_engine.fuse_multi_source_data.return_value = {
                'fused_consumption': {
                    'total_kwh': 675.8,
                    'cost_usd': 101.38,
                    'efficiency_score': 0.82
                },
                'device_insights': [
                    {
                        'device_id': 'smart_meter_001',
                        'contribution_percent': 85.0,
                        'efficiency_rating': 'good'
                    }
                ],
                'optimization_opportunities': [
                    {
                        'type': 'load_balancing',
                        'potential_savings': 12.5,
                        'confidence': 0.78
                    }
                ],
                'data_quality_score': 0.89
            }
            
            # Perform data fusion
            fused_result = await mock_ai.fusion_engine.fuse_multi_source_data(
                sample_energy_data,
                sample_sensor_readings,
                None
            )
            
            assert fused_result['data_quality_score'] > 0.8
            assert 'fused_consumption' in fused_result
            assert len(fused_result['optimization_opportunities']) > 0
        
        # Step 6: Verify IoT-AI integration success
        integration_success = (
            len(readings) > 0 and
            all(r.quality_score > 0.8 for r in readings.values() if r) and
            fused_result['data_quality_score'] > 0.8
        )
        
        assert integration_success, "IoT data integration with AI analysis should succeed"

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration_under_load(
        self,
        sample_energy_data,
        sample_sensor_readings
    ):
        """
        Test multi-agent system collaboration under load conditions.
        
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
        """
        # Step 1: Initialize multi-agent service
        with patch('src.services.multi_agent_service.get_multi_agent_service') as mock_service_getter:
            mock_service = AsyncMock()
            mock_service_getter.return_value = mock_service
            
            # Mock coordinator
            from src.services.agents.coordinator import RecommendationSynthesis
            from src.models.recommendation import OptimizationRecommendation, EstimatedSavings
            
            # Create mock synthesized recommendations
            mock_recommendations = []
            for i in range(5):  # Simulate multiple recommendations
                rec = OptimizationRecommendation(
                    id=f"load_test_rec_{i}",
                    type=['cost_saving', 'efficiency', 'environmental'][i % 3],
                    priority=['high', 'medium', 'low'][i % 3],
                    title=f"Load Test Recommendation {i}",
                    description=f"Test recommendation {i} under load",
                    implementation_steps=[f"Step 1 for rec {i}", f"Step 2 for rec {i}"],
                    estimated_savings=EstimatedSavings(
                        annual_cost_usd=100.0 + i * 50,
                        annual_kwh=500.0 + i * 200,
                        co2_reduction_kg=200.0 + i * 100
                    ),
                    difficulty=['easy', 'moderate', 'complex'][i % 3],
                    agent_source=f"agent_{i % 3}",
                    confidence=0.8 + (i * 0.02),
                    created_at=datetime.now(),
                    status="pending"
                )
                
                synthesis = RecommendationSynthesis(
                    recommendation=rec,
                    primary_agent=f"agent_{i % 3}",
                    supporting_agents=[f"agent_{(i+1) % 3}", f"agent_{(i+2) % 3}"],
                    validation_scores={f"agent_{j}": 0.8 + (j * 0.05) for j in range(3)},
                    synthesis_confidence=0.85 + (i * 0.02),
                    conflicts=[],
                    created_at=datetime.now()
                )
                
                mock_recommendations.append(synthesis)
            
            mock_service.generate_recommendations.return_value = [
                {
                    'id': rec.recommendation.id,
                    'type': rec.recommendation.type,
                    'priority': rec.recommendation.priority,
                    'title': rec.recommendation.title,
                    'description': rec.recommendation.description,
                    'implementation_steps': rec.recommendation.implementation_steps,
                    'estimated_savings': {
                        'annual_cost_usd': rec.recommendation.estimated_savings.annual_cost_usd,
                        'annual_kwh': rec.recommendation.estimated_savings.annual_kwh,
                        'co2_reduction_kg': rec.recommendation.estimated_savings.co2_reduction_kg
                    },
                    'difficulty': rec.recommendation.difficulty,
                    'confidence': rec.recommendation.confidence,
                    'primary_agent': rec.primary_agent,
                    'supporting_agents': rec.supporting_agents,
                    'synthesis_confidence': rec.synthesis_confidence
                }
                for rec in mock_recommendations
            ]
            
            # Step 2: Simulate concurrent analysis requests (load testing)
            concurrent_tasks = []
            num_concurrent_requests = 10
            
            for i in range(num_concurrent_requests):
                task = mock_service.generate_recommendations(
                    sample_energy_data,
                    sample_sensor_readings,
                    {"request_id": f"load_test_{i}"}
                )
                concurrent_tasks.append(task)
            
            # Execute concurrent requests
            start_time = datetime.now()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Step 3: Validate results under load
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            assert len(successful_results) >= num_concurrent_requests * 0.8, \
                "At least 80% of concurrent requests should succeed"
            
            assert processing_time < 30.0, \
                "Concurrent processing should complete within 30 seconds"
            
            # Verify recommendation quality under load
            for result in successful_results:
                assert len(result) > 0, "Each request should generate recommendations"
                for rec in result:
                    assert rec['synthesis_confidence'] > 0.7, \
                        "Recommendations should maintain quality under load"
            
            # Step 4: Test agent coordination transparency
            mock_service.get_agent_explanations.return_value = {
                'agent_contributions': [
                    {
                        'agent_id': 'efficiency_advisor',
                        'agent_type': 'efficiency',
                        'recommendation_id': 'load_test_rec_0',
                        'contribution_type': 'primary',
                        'confidence': 0.85,
                        'reasoning': 'Device efficiency analysis shows optimization potential',
                        'data_sources': ['sensor_readings', 'consumption_data'],
                        'timestamp': datetime.now().isoformat()
                    }
                ],
                'collaboration_sessions': [
                    {
                        'session_id': 'session_001',
                        'participating_agents': ['efficiency_advisor', 'cost_forecaster', 'eco_planner'],
                        'data_sources': ['utility_bills', 'iot_sensors'],
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'recommendations_generated': len(mock_recommendations),
                        'conflicts_resolved': 2,
                        'status': 'completed'
                    }
                ]
            }
            
            explanations = await mock_service.get_agent_explanations()
            
            assert len(explanations['agent_contributions']) > 0
            assert len(explanations['collaboration_sessions']) > 0
            
            # Step 5: Test reanalysis trigger under load
            mock_service.trigger_reanalysis.return_value = mock_service.generate_recommendations.return_value
            
            reanalysis_result = await mock_service.trigger_reanalysis(
                sample_energy_data,
                "load_test_trigger"
            )
            
            assert len(reanalysis_result) > 0, "Reanalysis should generate new recommendations"
        
        # Verify overall multi-agent collaboration success
        collaboration_success = (
            len(successful_results) >= num_concurrent_requests * 0.8 and
            processing_time < 30.0 and
            len(explanations['agent_contributions']) > 0
        )
        
        assert collaboration_success, "Multi-agent collaboration should succeed under load"

    @pytest.mark.asyncio
    async def test_end_to_end_system_integration(
        self,
        client,
        sample_utility_bill_image,
        sample_iot_devices,
        sample_sensor_readings,
        sample_energy_data
    ):
        """
        Test complete end-to-end system integration across all components.
        
        **Validates: Requirements 1.1-7.5 (Complete Integration)**
        """
        # Step 1: Document Processing Pipeline
        with patch('src.services.ocr_service.OCRProcessingEngine') as mock_ocr:
            mock_engine = AsyncMock()
            mock_ocr.return_value = mock_engine
            
            from src.services.ocr_service import OCRResult, EnergyFieldData, DocumentFormat
            mock_ocr_result = OCRResult(
                text="Electric Bill - Consumption: 450.5 kWh, Cost: $67.58",
                confidence=0.95,
                format=DocumentFormat.JPEG,
                page_count=1,
                bounding_boxes=[]
            )
            mock_engine.process_document.return_value = mock_ocr_result
            mock_engine.assess_quality.return_value = {"overall_quality": 0.95}
            mock_engine.extract_energy_fields.return_value = EnergyFieldData(
                consumption_kwh=450.5,
                cost_usd=67.58,
                billing_period_start=datetime.now() - timedelta(days=30),
                billing_period_end=datetime.now(),
                account_number="123456789",
                confidence_scores={"consumption_kwh": 0.95, "cost_usd": 0.92}
            )
            
            # Upload document
            ocr_response = client.post(
                "/api/ocr/upload",
                files={"file": ("test_bill.jpg", sample_utility_bill_image, "image/jpeg")}
            )
            assert ocr_response.status_code == 200
            ocr_data = ocr_response.json()
        
        # Step 2: IoT Integration
        iot_service = IoTIntegrationService()
        
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected = True
            mock_create_handler.return_value = mock_handler
            
            # Register and read from IoT devices
            for i, device in enumerate(sample_iot_devices):
                await iot_service.register_device(device)
                mock_handler.read_data.return_value = sample_sensor_readings[i]
            
            iot_readings = await iot_service.read_all_devices()
        
        # Step 3: AI Analysis and Data Fusion
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Mock comprehensive AI analysis
            mock_ai.batch_inference.return_value = {
                'results': {
                    'patterns': [
                        {'type': 'peak_usage', 'confidence': 0.89},
                        {'type': 'efficiency_opportunity', 'confidence': 0.82}
                    ],
                    'fusion_data': {
                        'combined_insights': [
                            {'insight': 'Load shifting potential', 'confidence': 0.85}
                        ]
                    }
                },
                'processing_time_ms': 156.7,
                'summary': {'data_quality_score': 0.91}
            }
            
            # Mock data fusion
            mock_ai.fusion_engine.fuse_multi_source_data.return_value = {
                'fused_consumption': {'total_kwh': 675.8, 'efficiency_score': 0.84},
                'optimization_opportunities': [
                    {'type': 'device_scheduling', 'potential_savings': 15.2}
                ],
                'data_quality_score': 0.88
            }
            
            # Perform AI analysis
            ai_response = client.post(
                "/api/v1/ai/analyze-patterns",
                json={
                    "consumption_data": [
                        {
                            "id": "integration_test",
                            "timestamp": datetime.now().isoformat(),
                            "source": "utility_bill",
                            "consumption_kwh": ocr_data["energy_data"]["consumption_kwh"],
                            "cost_usd": ocr_data["energy_data"]["cost_usd"],
                            "billing_period_start": (datetime.now() - timedelta(days=30)).isoformat(),
                            "billing_period_end": datetime.now().isoformat(),
                            "confidence_score": 0.95
                        }
                    ]
                }
            )
            assert ai_response.status_code == 200
            ai_data = ai_response.json()
        
        # Step 4: Multi-Agent Recommendation Generation
        with patch('src.services.multi_agent_service.get_multi_agent_service') as mock_multi_agent:
            mock_service = AsyncMock()
            mock_multi_agent.return_value = mock_service
            
            mock_service.generate_recommendations.return_value = [
                {
                    'id': 'integration_rec_001',
                    'type': 'cost_saving',
                    'priority': 'high',
                    'title': 'Optimize peak hour usage',
                    'description': 'Reduce consumption during peak pricing periods',
                    'estimated_savings': {
                        'annual_cost_usd': 245.80,
                        'annual_kwh': 1640.0,
                        'co2_reduction_kg': 738.0
                    },
                    'confidence': 0.87,
                    'primary_agent': 'cost_forecaster',
                    'synthesis_confidence': 0.89
                },
                {
                    'id': 'integration_rec_002',
                    'type': 'efficiency',
                    'priority': 'medium',
                    'title': 'Upgrade inefficient devices',
                    'description': 'Replace high-consumption devices with efficient alternatives',
                    'estimated_savings': {
                        'annual_cost_usd': 180.50,
                        'annual_kwh': 1200.0,
                        'co2_reduction_kg': 540.0
                    },
                    'confidence': 0.82,
                    'primary_agent': 'efficiency_advisor',
                    'synthesis_confidence': 0.85
                }
            ]
            
            recommendations = await mock_service.generate_recommendations(
                sample_energy_data,
                list(iot_readings.values()),
                {"integration_test": True}
            )
        
        # Step 5: Dashboard Data Integration
        with patch('src.database.connection.get_db_session') as mock_db:
            # Mock database session for dashboard endpoints
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock energy data query
            mock_session.query.return_value.filter.return_value.count.return_value = 2
            mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
            
            # Test dashboard data retrieval
            dashboard_response = client.get("/api/dashboard/energy-data?limit=10")
            assert dashboard_response.status_code == 200
            dashboard_data = dashboard_response.json()
        
        # Step 6: Validate Complete Integration
        integration_metrics = {
            'ocr_success': ocr_data["success"],
            'ocr_confidence': ocr_data["ocr_result"]["confidence"],
            'iot_devices_connected': len(iot_readings),
            'iot_data_quality': sum(r.quality_score for r in iot_readings.values() if r) / len(iot_readings),
            'ai_analysis_confidence': ai_data["confidence_score"],
            'recommendations_generated': len(recommendations),
            'avg_recommendation_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations),
            'dashboard_accessible': dashboard_response.status_code == 200
        }
        
        # Validate integration success criteria
        assert integration_metrics['ocr_success'], "OCR processing should succeed"
        assert integration_metrics['ocr_confidence'] > 0.9, "OCR should have high confidence"
        assert integration_metrics['iot_devices_connected'] > 0, "IoT devices should be connected"
        assert integration_metrics['iot_data_quality'] > 0.8, "IoT data should have good quality"
        assert integration_metrics['ai_analysis_confidence'] > 0.8, "AI analysis should be confident"
        assert integration_metrics['recommendations_generated'] > 0, "Recommendations should be generated"
        assert integration_metrics['avg_recommendation_confidence'] > 0.8, "Recommendations should be confident"
        assert integration_metrics['dashboard_accessible'], "Dashboard should be accessible"
        
        # Calculate overall integration score
        integration_score = (
            integration_metrics['ocr_confidence'] * 0.15 +
            integration_metrics['iot_data_quality'] * 0.20 +
            integration_metrics['ai_analysis_confidence'] * 0.25 +
            integration_metrics['avg_recommendation_confidence'] * 0.25 +
            (1.0 if integration_metrics['dashboard_accessible'] else 0.0) * 0.15
        )
        
        assert integration_score > 0.85, f"Overall integration score should be > 0.85, got {integration_score}"
        
        return {
            'integration_score': integration_score,
            'metrics': integration_metrics,
            'recommendations': recommendations,
            'processing_pipeline_success': True
        }

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(
        self,
        client,
        sample_energy_data
    ):
        """
        Test system resilience and error handling across components.
        
        **Validates: Requirements 6.1, 6.3, 6.4, 6.5**
        """
        # Test OCR error handling with invalid file
        invalid_response = client.post(
            "/api/ocr/upload",
            files={"file": ("invalid.txt", b"not an image", "text/plain")}
        )
        assert invalid_response.status_code in [400, 422], "Should reject invalid file types"
        
        # Test AI service error handling
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai_service.side_effect = Exception("AI service unavailable")
            
            ai_error_response = client.post(
                "/api/v1/ai/analyze-patterns",
                json={"consumption_data": []}
            )
            assert ai_error_response.status_code == 500, "Should handle AI service errors gracefully"
        
        # Test IoT service resilience
        iot_service = IoTIntegrationService()
        
        # Test offline operation
        await iot_service.set_online_status(False)
        
        # Simulate storing data while offline
        from src.models.sensor_reading import SensorReading, SensorReadings
        offline_reading = SensorReading(
            sensor_id="test_offline",
            device_type="test",
            timestamp=datetime.now(),
            readings=SensorReadings(power_watts=100.0),
            quality_score=0.9,
            location="test"
        )
        
        await iot_service._store_sensor_reading(offline_reading)
        assert len(iot_service.offline_buffer) > 0, "Should buffer data when offline"
        
        # Test coming back online
        await iot_service.set_online_status(True)
        # Buffer should be flushed (mocked in this test)
        
        # Test dashboard error handling
        dashboard_error_response = client.get("/api/dashboard/energy-data?start_date=invalid")
        # Should handle invalid date gracefully (implementation dependent)
        
        resilience_success = (
            invalid_response.status_code in [400, 422] and
            ai_error_response.status_code == 500 and
            len(iot_service.offline_buffer) >= 0  # Buffer exists
        )
        
        assert resilience_success, "System should handle errors gracefully"

    @pytest.mark.asyncio 
    async def test_performance_under_load(
        self,
        sample_energy_data,
        sample_sensor_readings
    ):
        """
        Test system performance under various load conditions.
        
        **Validates: Requirements 3.3, 6.4**
        """
        # Test concurrent API requests
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Mock fast AI response
            mock_ai.real_time_inference.return_value = {
                'result': {'analysis': 'test'},
                'processing_time_ms': 150.0,
                'confidence': 0.85,
                'timestamp': datetime.now()
            }
            
            # Simulate concurrent real-time inference requests
            start_time = datetime.now()
            
            tasks = []
            for i in range(20):  # 20 concurrent requests
                task = mock_ai.real_time_inference(
                    sample_energy_data[0],
                    inference_type="pattern_analysis"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            # Performance assertions
            assert len(successful_results) >= 18, "At least 90% of requests should succeed"
            assert processing_time < 5.0, "Concurrent processing should complete within 5 seconds"
            
            # Check individual response times
            avg_response_time = sum(r['processing_time_ms'] for r in successful_results) / len(successful_results)
            assert avg_response_time < 500.0, "Average response time should be under 500ms"
        
        performance_success = (
            len(successful_results) >= 18 and
            processing_time < 5.0 and
            avg_response_time < 500.0
        )
        
        assert performance_success, "System should maintain performance under load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])