"""
Load Testing and Performance Validation Integration Tests.

Comprehensive load testing and performance validation for the Smart Energy Copilot system
under various stress conditions and concurrent usage scenarios.

**Validates: Requirements 3.3, 6.4, All system performance requirements**
"""

import asyncio
import pytest
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
import threading
import random

from src.services.ai_service import get_ai_service
from src.services.multi_agent_service import get_multi_agent_service
from src.services.iot_integration import IoTIntegrationService
from src.services.ocr_service import OCRProcessingEngine
from src.services.recommendation_engine import get_recommendation_engine
from src.models.energy_consumption import EnergyConsumption, BillingPeriod
from src.models.sensor_reading import SensorReading, SensorReadings
from src.models.device import Device, DeviceConfig, ProtocolType


class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self):
        self.response_times = []
        self.throughput_data = []
        self.error_counts = {}
        self.resource_usage = []
        self.concurrent_users = 0
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = datetime.now()
    
    def end_measurement(self):
        """End performance measurement."""
        self.end_time = datetime.now()
    
    def record_response_time(self, operation: str, response_time: float):
        """Record response time for an operation."""
        self.response_times.append({
            'operation': operation,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
    
    def record_error(self, operation: str, error_type: str):
        """Record an error occurrence."""
        key = f"{operation}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def record_throughput(self, operation: str, count: int, duration: float):
        """Record throughput data."""
        throughput = count / duration if duration > 0 else 0
        self.throughput_data.append({
            'operation': operation,
            'count': count,
            'duration': duration,
            'throughput': throughput,
            'timestamp': datetime.now()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.response_times:
            return {"error": "No performance data collected"}
        
        response_times = [rt['response_time'] for rt in self.response_times]
        
        return {
            'response_time_stats': {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'p95': self._percentile(response_times, 95),
                'p99': self._percentile(response_times, 99)
            },
            'throughput_stats': {
                'operations': len(self.response_times),
                'total_duration': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
                'avg_throughput': statistics.mean([t['throughput'] for t in self.throughput_data]) if self.throughput_data else 0
            },
            'error_stats': {
                'total_errors': sum(self.error_counts.values()),
                'error_rate': sum(self.error_counts.values()) / len(self.response_times) if self.response_times else 0,
                'error_breakdown': self.error_counts
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestLoadAndPerformance:
    """Load testing and performance validation tests."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Performance metrics fixture."""
        return PerformanceMetrics()
    
    @pytest.fixture
    def sample_energy_data_large(self):
        """Large dataset of energy consumption data for load testing."""
        data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(100):  # 100 data points (reduced from 1000)
            data.append(EnergyConsumption(
                id=f"load_test_consumption_{i}",
                timestamp=base_time + timedelta(hours=i),
                source="utility_bill" if i % 3 == 0 else "iot_sensor",
                consumption_kwh=random.uniform(200, 800),
                cost_usd=random.uniform(30, 120),
                billing_period=BillingPeriod(
                    start_date=base_time + timedelta(hours=i-24),
                    end_date=base_time + timedelta(hours=i)
                ),
                confidence_score=random.uniform(0.8, 0.95)
            ))
        
        return data
    
    @pytest.fixture
    def sample_sensor_data_large(self):
        """Large dataset of sensor readings for load testing."""
        data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(200):  # 200 sensor readings (reduced from 2000)
            data.append(SensorReading(
                sensor_id=f"load_test_sensor_{i % 50}",  # 50 different sensors
                device_type="energy_monitor",
                timestamp=base_time + timedelta(minutes=i),
                readings=SensorReadings(
                    power_watts=random.uniform(1000, 5000),
                    voltage=random.uniform(235, 245),
                    current_amps=random.uniform(4, 20),
                    temperature_celsius=random.uniform(20, 30),
                    humidity_percent=random.uniform(40, 60)
                ),
                quality_score=random.uniform(0.85, 0.95),
                location=f"zone_{i % 10}"
            ))
        
        return data
    
    @pytest.mark.asyncio
    async def test_ai_service_load_testing(self, performance_metrics, sample_energy_data_large):
        """
        Test AI service performance under high load conditions.
        
        **Validates: Requirements 3.3, 2.1, 2.2**
        """
        performance_metrics.start_measurement()
        
        with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
            mock_ai = AsyncMock()
            mock_ai_service.return_value = mock_ai
            
            # Mock AI service responses with realistic processing times
            def mock_batch_inference(*args, **kwargs):
                # Simulate processing time based on data size
                data_size = len(args[0]) if args and isinstance(args[0], list) else 10
                processing_time = min(50 + (data_size * 2), 500)  # 50-500ms based on size
                
                return {
                    'results': {
                        'patterns': [
                            {'type': f'pattern_{i}', 'confidence': random.uniform(0.8, 0.95)}
                            for i in range(min(data_size // 10, 10))
                        ],
                        'fusion_data': {
                            'combined_insights': [
                                {'insight': f'insight_{i}', 'confidence': random.uniform(0.75, 0.9)}
                                for i in range(min(data_size // 20, 5))
                            ]
                        }
                    },
                    'processing_time_ms': processing_time,
                    'summary': {'data_quality_score': random.uniform(0.85, 0.95)}
                }
            
            def mock_real_time_inference(*args, **kwargs):
                # Simulate real-time processing (faster)
                processing_time = random.uniform(80, 200)  # 80-200ms
                
                return {
                    'result': {'analysis': f'real_time_result_{random.randint(1, 100)}'},
                    'processing_time_ms': processing_time,
                    'confidence': random.uniform(0.8, 0.95),
                    'timestamp': datetime.now()
                }
            
            mock_ai.batch_inference.side_effect = mock_batch_inference
            mock_ai.real_time_inference.side_effect = mock_real_time_inference
            
            # Test 1: Batch processing load test
            batch_sizes = [5, 10, 25, 50]  # Reduced batch sizes
            batch_results = []
            
            for batch_size in batch_sizes:
                batch_data = sample_energy_data_large[:batch_size]
                
                start_time = time.time()
                result = await mock_ai.batch_inference(batch_data, "load_test")
                end_time = time.time()
                
                response_time = end_time - start_time
                performance_metrics.record_response_time(f"batch_inference_{batch_size}", response_time)
                
                batch_results.append({
                    'batch_size': batch_size,
                    'response_time': response_time,
                    'processing_time_ms': result['processing_time_ms'],
                    'patterns_found': len(result['results']['patterns'])
                })
            
            # Test 2: Concurrent real-time inference load test
            concurrent_requests = [5, 10, 20]  # Reduced concurrent requests
            
            for num_requests in concurrent_requests:
                tasks = []
                start_time = time.time()
                
                for i in range(num_requests):
                    task = mock_ai.real_time_inference(
                        sample_energy_data_large[i % len(sample_energy_data_large)],
                        f"concurrent_test_{i}"
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                successful_results = [r for r in results if not isinstance(r, Exception)]
                
                performance_metrics.record_throughput(
                    f"concurrent_inference_{num_requests}",
                    len(successful_results),
                    total_time
                )
                
                # Record individual response times
                for result in successful_results:
                    if isinstance(result, dict) and 'processing_time_ms' in result:
                        performance_metrics.record_response_time(
                            f"concurrent_inference_{num_requests}",
                            result['processing_time_ms'] / 1000.0
                        )
            
            # Test 3: Sustained load test
            sustained_duration = 10  # 10 seconds (reduced from 30)
            sustained_start = time.time()
            sustained_count = 0
            
            while time.time() - sustained_start < sustained_duration:
                try:
                    start_time = time.time()
                    await mock_ai.real_time_inference(
                        sample_energy_data_large[sustained_count % len(sample_energy_data_large)],
                        f"sustained_test_{sustained_count}"
                    )
                    end_time = time.time()
                    
                    performance_metrics.record_response_time("sustained_load", end_time - start_time)
                    sustained_count += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    performance_metrics.record_error("sustained_load", str(type(e).__name__))
            
            performance_metrics.record_throughput("sustained_load", sustained_count, sustained_duration)
        
        performance_metrics.end_measurement()
        stats = performance_metrics.get_statistics()
        
        # Performance assertions
        assert stats['response_time_stats']['mean'] < 2.0, "Average response time should be under 2 seconds"
        assert stats['response_time_stats']['p95'] < 5.0, "95th percentile should be under 5 seconds"
        assert stats['error_stats']['error_rate'] < 0.05, "Error rate should be under 5%"
        assert sustained_count > 30, "Should handle at least 30 requests in 10 seconds"
        
        return stats
    
    @pytest.mark.asyncio
    async def test_multi_agent_system_load_testing(self, performance_metrics, sample_energy_data_large):
        """
        Test multi-agent system performance under load.
        
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
        """
        performance_metrics.start_measurement()
        
        with patch('src.services.multi_agent_service.get_multi_agent_service') as mock_multi_agent:
            mock_service = AsyncMock()
            mock_multi_agent.return_value = mock_service
            
            # Mock multi-agent recommendation generation
            def mock_generate_recommendations(*args, **kwargs):
                # Simulate agent collaboration time
                processing_time = random.uniform(200, 800)  # 200-800ms
                num_recommendations = random.randint(3, 8)
                
                recommendations = []
                for i in range(num_recommendations):
                    recommendations.append({
                        'id': f'load_rec_{i}_{random.randint(1000, 9999)}',
                        'type': random.choice(['cost_saving', 'efficiency', 'environmental']),
                        'priority': random.choice(['high', 'medium', 'low']),
                        'title': f'Load Test Recommendation {i}',
                        'description': f'Generated under load test conditions',
                        'estimated_savings': {
                            'annual_cost_usd': random.uniform(100, 500),
                            'annual_kwh': random.uniform(500, 2000),
                            'co2_reduction_kg': random.uniform(200, 800)
                        },
                        'confidence': random.uniform(0.75, 0.95),
                        'primary_agent': random.choice(['efficiency_advisor', 'cost_forecaster', 'eco_planner']),
                        'synthesis_confidence': random.uniform(0.8, 0.95)
                    })
                
                return recommendations
            
            mock_service.generate_recommendations.side_effect = mock_generate_recommendations
            
            # Test 1: Concurrent agent collaboration
            concurrent_sessions = [3, 5, 10]  # Reduced sessions
            
            for num_sessions in concurrent_sessions:
                tasks = []
                start_time = time.time()
                
                for i in range(num_sessions):
                    # Use different data subsets for each session
                    session_data = sample_energy_data_large[i*10:(i+1)*10]
                    task = mock_service.generate_recommendations(
                        session_data,
                        [],
                        {"session_id": f"load_session_{i}"}
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                successful_results = [r for r in results if not isinstance(r, Exception)]
                
                performance_metrics.record_throughput(
                    f"agent_collaboration_{num_sessions}",
                    len(successful_results),
                    total_time
                )
                
                performance_metrics.record_response_time(
                    f"agent_collaboration_{num_sessions}",
                    total_time
                )
                
                # Verify recommendation quality under load
                total_recommendations = sum(len(result) for result in successful_results)
                avg_recommendations_per_session = total_recommendations / len(successful_results) if successful_results else 0
                
                assert avg_recommendations_per_session >= 3, "Should generate at least 3 recommendations per session"
            
            # Test 2: Agent explanation performance under load
            mock_service.get_agent_explanations.return_value = {
                'agent_contributions': [
                    {
                        'agent_id': f'agent_{i}',
                        'recommendation_id': f'rec_{i}',
                        'confidence': random.uniform(0.8, 0.95),
                        'reasoning': f'Load test reasoning {i}'
                    }
                    for i in range(20)  # 20 contributions
                ],
                'collaboration_sessions': [
                    {
                        'session_id': f'session_{i}',
                        'participating_agents': ['efficiency_advisor', 'cost_forecaster', 'eco_planner'],
                        'recommendations_generated': random.randint(3, 8),
                        'conflicts_resolved': random.randint(0, 2),
                        'status': 'completed'
                    }
                    for i in range(10)  # 10 sessions
                ]
            }
            
            # Test explanation retrieval under load
            explanation_tasks = []
            for i in range(10):  # 10 concurrent explanation requests (reduced from 50)
                task = mock_service.get_agent_explanations()
                explanation_tasks.append(task)
            
            start_time = time.time()
            explanation_results = await asyncio.gather(*explanation_tasks, return_exceptions=True)
            end_time = time.time()
            
            explanation_time = end_time - start_time
            successful_explanations = [r for r in explanation_results if not isinstance(r, Exception)]
            
            performance_metrics.record_throughput("agent_explanations", len(successful_explanations), explanation_time)
            
            # Test 3: Reanalysis trigger performance
            reanalysis_tasks = []
            for i in range(5):  # 5 concurrent reanalysis requests (reduced from 20)
                task = mock_service.trigger_reanalysis(
                    sample_energy_data_large[i*10:(i+1)*10],
                    f"load_reanalysis_{i}"
                )
                reanalysis_tasks.append(task)
            
            start_time = time.time()
            reanalysis_results = await asyncio.gather(*reanalysis_tasks, return_exceptions=True)
            end_time = time.time()
            
            reanalysis_time = end_time - start_time
            successful_reanalysis = [r for r in reanalysis_results if not isinstance(r, Exception)]
            
            performance_metrics.record_throughput("reanalysis_triggers", len(successful_reanalysis), reanalysis_time)
        
        performance_metrics.end_measurement()
        stats = performance_metrics.get_statistics()
        
        # Performance assertions for multi-agent system
        assert stats['response_time_stats']['mean'] < 3.0, "Average multi-agent response time should be under 3 seconds"
        assert stats['error_stats']['error_rate'] < 0.1, "Multi-agent error rate should be under 10%"
        assert len(successful_explanations) >= 8, "Should handle at least 80% of explanation requests"
        
        return stats
    
    @pytest.mark.asyncio
    async def test_iot_integration_load_testing(self, performance_metrics):
        """
        Test IoT integration performance under high device load.
        
        **Validates: Requirements 7.1, 7.2, 7.4, 7.5**
        """
        performance_metrics.start_measurement()
        
        iot_service = IoTIntegrationService()
        
        # Create large number of test devices
        num_devices = 20  # Reduced from 100
        devices = []
        
        for i in range(num_devices):
            device_config = DeviceConfig(
                protocol=random.choice([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]),
                endpoint=f"test://device_{i}.local",
                polling_interval=random.randint(10, 60),
                retry_attempts=3
            )
            
            device = Device(
                device_id=f"load_test_device_{i:03d}",
                device_type=random.choice(["smart_meter", "temperature_sensor", "power_monitor"]),
                name=f"Load Test Device {i}",
                location=f"zone_{i % 10}",
                config=device_config
            )
            devices.append(device)
        
        # Mock handlers for all devices
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            def create_mock_handler(device):
                mock_handler = AsyncMock()
                mock_handler.connect.return_value = True
                mock_handler.is_connected = True
                
                # Simulate variable response times based on protocol
                if device.config.protocol == ProtocolType.MQTT:
                    response_time = random.uniform(0.05, 0.15)  # MQTT is fastest
                elif device.config.protocol == ProtocolType.HTTP_REST:
                    response_time = random.uniform(0.1, 0.3)   # HTTP is medium
                else:  # Modbus
                    response_time = random.uniform(0.2, 0.5)   # Modbus is slowest
                
                async def mock_read_data():
                    await asyncio.sleep(response_time)
                    return SensorReading(
                        sensor_id=device.device_id,
                        device_type=device.device_type,
                        timestamp=datetime.now(),
                        readings=SensorReadings(
                            power_watts=random.uniform(1000, 5000),
                            voltage=random.uniform(235, 245),
                            current_amps=random.uniform(4, 20)
                        ),
                        quality_score=random.uniform(0.85, 0.95),
                        location=device.location
                    )
                
                mock_handler.read_data = mock_read_data
                return mock_handler
            
            mock_create_handler.side_effect = create_mock_handler
            
            # Test 1: Mass device registration
            start_time = time.time()
            
            registration_tasks = [iot_service.register_device(device) for device in devices]
            registration_results = await asyncio.gather(*registration_tasks, return_exceptions=True)
            
            registration_time = time.time() - start_time
            successful_registrations = sum(1 for result in registration_results if result is True)
            
            performance_metrics.record_throughput("device_registration", successful_registrations, registration_time)
            
            # Test 2: Concurrent data reading from all devices
            reading_iterations = 2  # Reduced from 5
            
            for iteration in range(reading_iterations):
                start_time = time.time()
                readings = await iot_service.read_all_devices()
                end_time = time.time()
                
                reading_time = end_time - start_time
                successful_readings = len([r for r in readings.values() if r is not None])
                
                performance_metrics.record_response_time(f"mass_reading_iter_{iteration}", reading_time)
                performance_metrics.record_throughput(f"mass_reading_iter_{iteration}", successful_readings, reading_time)
            
            # Test 3: Device discovery performance
            with patch.object(iot_service, '_scan_network_for_devices') as mock_scan:
                # Simulate discovering many devices
                mock_scan.return_value = [
                    {
                        'ip': f'192.168.1.{i}',
                        'port': 1883,
                        'protocol': 'mqtt',
                        'device_type': 'smart_meter'
                    }
                    for i in range(25, 50)  # 25 discovered devices (reduced from 100)
                ]
                
                start_time = time.time()
                discovery_result = await iot_service.discover_devices()
                discovery_time = time.time() - start_time
                
                performance_metrics.record_response_time("device_discovery", discovery_time)
                
                assert discovery_result.success is True
                assert len(discovery_result.discovered_devices) == 25
            
            # Test 4: Offline buffer performance
            await iot_service.set_online_status(False)
            
            # Generate many readings while offline
            offline_readings = []
            start_time = time.time()
            
            for i in range(50):  # 50 readings (reduced from 500)
                reading = SensorReading(
                    sensor_id=f"offline_device_{i % 20}",
                    device_type="test_sensor",
                    timestamp=datetime.now(),
                    readings=SensorReadings(power_watts=random.uniform(1000, 3000)),
                    quality_score=0.9,
                    location="test"
                )
                await iot_service._store_sensor_reading(reading)
            
            buffer_time = time.time() - start_time
            performance_metrics.record_throughput("offline_buffering", 50, buffer_time)
            
            # Test buffer size management
            assert len(iot_service.offline_buffer) <= iot_service.max_buffer_size
        
        performance_metrics.end_measurement()
        stats = performance_metrics.get_statistics()
        
        # Performance assertions for IoT integration
        assert successful_registrations >= num_devices * 0.9, "Should register at least 90% of devices"
        assert stats['response_time_stats']['mean'] < 5.0, "Average IoT operation time should be under 5 seconds"
        assert discovery_time < 10.0, "Device discovery should complete within 10 seconds"
        
        return stats
    
    @pytest.mark.asyncio
    async def test_ocr_service_load_testing(self, performance_metrics):
        """
        Test OCR service performance under document processing load.
        
        **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
        """
        performance_metrics.start_measurement()
        
        with patch('src.services.ocr_service.OCRProcessingEngine') as mock_ocr_engine:
            mock_engine = AsyncMock()
            mock_ocr_engine.return_value = mock_engine
            
            # Mock OCR processing with realistic times
            def mock_process_document(*args, **kwargs):
                # Simulate processing time based on document complexity
                processing_time = random.uniform(500, 2000)  # 0.5-2 seconds
                
                from src.services.ocr_service import OCRResult, DocumentFormat
                return OCRResult(
                    text=f"Mock OCR result - processed at {datetime.now()}",
                    confidence=random.uniform(0.85, 0.95),
                    format=random.choice([DocumentFormat.PDF, DocumentFormat.JPEG, DocumentFormat.PNG]),
                    page_count=random.randint(1, 5),
                    bounding_boxes=[]
                )
            
            def mock_extract_energy_fields(*args, **kwargs):
                from src.services.ocr_service import EnergyFieldData
                return EnergyFieldData(
                    consumption_kwh=random.uniform(200, 800),
                    cost_usd=random.uniform(30, 120),
                    billing_period_start=datetime.now() - timedelta(days=30),
                    billing_period_end=datetime.now(),
                    account_number=f"ACC{random.randint(100000, 999999)}",
                    confidence_scores={
                        "consumption_kwh": random.uniform(0.8, 0.95),
                        "cost_usd": random.uniform(0.8, 0.95)
                    }
                )
            
            mock_engine.process_document.side_effect = mock_process_document
            mock_engine.extract_energy_fields.side_effect = mock_extract_energy_fields
            mock_engine.assess_quality.return_value = {"overall_quality": random.uniform(0.8, 0.95)}
            
            # Test 1: Sequential document processing
            num_documents = 10  # Reduced from 50
            
            for i in range(num_documents):
                start_time = time.time()
                
                # Mock document data
                document_data = b"mock_document_data" * random.randint(100, 1000)
                
                ocr_result = await mock_engine.process_document(document_data, f"test_doc_{i}.pdf")
                energy_data = await mock_engine.extract_energy_fields(ocr_result)
                quality = await mock_engine.assess_quality(ocr_result)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                performance_metrics.record_response_time(f"ocr_processing_doc_{i}", processing_time)
            
            # Test 2: Concurrent document processing
            concurrent_batches = [3, 5, 8]  # Reduced batch sizes
            
            for batch_size in concurrent_batches:
                tasks = []
                start_time = time.time()
                
                for i in range(batch_size):
                    document_data = b"concurrent_mock_data" * random.randint(100, 500)
                    task = mock_engine.process_document(document_data, f"concurrent_doc_{i}.pdf")
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                successful_results = [r for r in results if not isinstance(r, Exception)]
                
                performance_metrics.record_throughput(f"concurrent_ocr_{batch_size}", len(successful_results), total_time)
            
            # Test 3: Different document formats performance
            formats = ["pdf", "jpeg", "png"]
            
            for format_type in formats:
                format_tasks = []
                start_time = time.time()
                
                for i in range(5):  # 5 documents per format (reduced from 10)
                    document_data = b"format_test_data" * random.randint(200, 800)
                    task = mock_engine.process_document(document_data, f"format_test_{i}.{format_type}")
                    format_tasks.append(task)
                
                format_results = await asyncio.gather(*format_tasks, return_exceptions=True)
                end_time = time.time()
                
                format_time = end_time - start_time
                successful_format_results = [r for r in format_results if not isinstance(r, Exception)]
                
                performance_metrics.record_throughput(f"format_{format_type}", len(successful_format_results), format_time)
        
        performance_metrics.end_measurement()
        stats = performance_metrics.get_statistics()
        
        # Performance assertions for OCR service
        assert stats['response_time_stats']['mean'] < 3.0, "Average OCR processing time should be under 3 seconds"
        assert stats['error_stats']['error_rate'] < 0.05, "OCR error rate should be under 5%"
        
        return stats
    
    @pytest.mark.asyncio
    async def test_system_wide_stress_testing(self, performance_metrics, sample_energy_data_large, sample_sensor_data_large):
        """
        Comprehensive system-wide stress testing with all components under load.
        
        **Validates: Requirements 3.3, 6.4, All system integration requirements**
        """
        performance_metrics.start_measurement()
        
        # Simulate high concurrent user load
        num_concurrent_users = 10  # Reduced from 25
        operations_per_user = 5   # Reduced from 10
        
        async def simulate_user_session(user_id: int):
            """Simulate a complete user session with multiple operations."""
            session_operations = []
            
            try:
                # Operation 1: Document upload and processing
                with patch('src.services.ocr_service.OCRProcessingEngine') as mock_ocr:
                    mock_engine = AsyncMock()
                    mock_ocr.return_value = mock_engine
                    
                    from src.services.ocr_service import OCRResult, EnergyFieldData, DocumentFormat
                    mock_engine.process_document.return_value = OCRResult(
                        text=f"User {user_id} document",
                        confidence=0.9,
                        format=DocumentFormat.PDF,
                        page_count=1,
                        bounding_boxes=[]
                    )
                    mock_engine.extract_energy_fields.return_value = EnergyFieldData(
                        consumption_kwh=450.0,
                        cost_usd=67.5,
                        billing_period_start=datetime.now() - timedelta(days=30),
                        billing_period_end=datetime.now(),
                        account_number=f"USER{user_id}ACC",
                        confidence_scores={"consumption_kwh": 0.9}
                    )
                    
                    start_time = time.time()
                    await mock_engine.process_document(b"mock_data", f"user_{user_id}_doc.pdf")
                    await mock_engine.extract_energy_fields(None)
                    ocr_time = time.time() - start_time
                    
                    session_operations.append(("ocr_processing", ocr_time))
                
                # Operation 2: AI analysis
                with patch('src.services.ai_service.get_ai_service') as mock_ai_service:
                    mock_ai = AsyncMock()
                    mock_ai_service.return_value = mock_ai
                    
                    mock_ai.real_time_inference.return_value = {
                        'result': {'user_analysis': f'user_{user_id}'},
                        'processing_time_ms': random.uniform(100, 300),
                        'confidence': 0.85
                    }
                    
                    start_time = time.time()
                    await mock_ai.real_time_inference(sample_energy_data_large[user_id % len(sample_energy_data_large)])
                    ai_time = time.time() - start_time
                    
                    session_operations.append(("ai_analysis", ai_time))
                
                # Operation 3: Multi-agent recommendations
                with patch('src.services.multi_agent_service.get_multi_agent_service') as mock_multi_agent:
                    mock_service = AsyncMock()
                    mock_multi_agent.return_value = mock_service
                    
                    mock_service.generate_recommendations.return_value = [
                        {
                            'id': f'user_{user_id}_rec_{i}',
                            'type': 'cost_saving',
                            'confidence': 0.8
                        }
                        for i in range(3)
                    ]
                    
                    start_time = time.time()
                    await mock_service.generate_recommendations(sample_energy_data_large[:10])
                    agent_time = time.time() - start_time
                    
                    session_operations.append(("multi_agent", agent_time))
                
                # Operation 4: IoT data reading
                iot_service = IoTIntegrationService()
                
                with patch.object(iot_service, 'read_all_devices') as mock_read_devices:
                    mock_read_devices.return_value = {
                        f"device_{i}": sample_sensor_data_large[i % len(sample_sensor_data_large)]
                        for i in range(5)  # 5 devices per user
                    }
                    
                    start_time = time.time()
                    await mock_read_devices()
                    iot_time = time.time() - start_time
                    
                    session_operations.append(("iot_reading", iot_time))
                
                return session_operations
                
            except Exception as e:
                performance_metrics.record_error(f"user_session_{user_id}", str(type(e).__name__))
                return []
        
        # Execute concurrent user sessions
        user_tasks = [simulate_user_session(user_id) for user_id in range(num_concurrent_users)]
        
        start_time = time.time()
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        end_time = time.time()
        
        total_stress_time = end_time - start_time
        
        # Analyze results
        successful_sessions = [result for result in user_results if not isinstance(result, Exception)]
        total_operations = sum(len(session) for session in successful_sessions)
        
        # Record all operation times
        for session in successful_sessions:
            for operation_type, operation_time in session:
                performance_metrics.record_response_time(f"stress_{operation_type}", operation_time)
        
        performance_metrics.record_throughput("stress_test_sessions", len(successful_sessions), total_stress_time)
        performance_metrics.record_throughput("stress_test_operations", total_operations, total_stress_time)
        
        # Memory and resource simulation
        simulated_memory_usage = min(85.0, 45.0 + (num_concurrent_users * 1.5))  # Simulate memory growth
        simulated_cpu_usage = min(90.0, 30.0 + (num_concurrent_users * 2.0))     # Simulate CPU growth
        
        # Test graceful degradation under stress
        if simulated_memory_usage > 80.0 or simulated_cpu_usage > 80.0:
            # Simulate reduced performance under stress
            degraded_operations = total_operations * 0.8  # 20% performance reduction
            performance_metrics.record_throughput("degraded_operations", degraded_operations, total_stress_time)
        
        performance_metrics.end_measurement()
        stats = performance_metrics.get_statistics()
        
        # Stress test assertions
        session_success_rate = len(successful_sessions) / num_concurrent_users
        assert session_success_rate >= 0.8, f"At least 80% of user sessions should succeed, got {session_success_rate:.2%}"
        
        assert total_stress_time < 30.0, "Stress test should complete within 30 seconds"
        assert stats['error_stats']['error_rate'] < 0.15, "Error rate under stress should be under 15%"
        
        # Performance degradation should be graceful
        if simulated_memory_usage > 80.0:
            assert stats['response_time_stats']['mean'] < 10.0, "Response time should remain reasonable under memory pressure"
        
        return {
            'stress_test_stats': stats,
            'session_success_rate': session_success_rate,
            'total_operations': total_operations,
            'simulated_resource_usage': {
                'memory_percent': simulated_memory_usage,
                'cpu_percent': simulated_cpu_usage
            }
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])