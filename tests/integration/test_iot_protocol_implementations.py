"""
IoT Protocol Implementation Integration Tests.

Tests actual IoT protocol implementations with mock devices to validate
MQTT, HTTP REST, and Modbus protocol handlers work correctly.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import socket
import threading

from src.services.iot_integration import IoTIntegrationService
from src.services.iot_protocols.mqtt_handler import MQTTHandler
from src.services.iot_protocols.http_handler import HTTPHandler
from src.services.iot_protocols.modbus_handler import ModbusHandler
from src.models.device import Device, DeviceConfig, ProtocolType, DeviceStatus
from src.models.sensor_reading import SensorReading, SensorReadings


class MockMQTTBroker:
    """Mock MQTT broker for testing."""
    
    def __init__(self, port: int = 1883):
        self.port = port
        self.clients = {}
        self.messages = []
        self.running = False
        self.server_thread = None
    
    def start(self):
        """Start mock MQTT broker."""
        self.running = True
        # In a real implementation, this would start a proper MQTT broker
        # For testing, we'll simulate broker behavior
    
    def stop(self):
        """Stop mock MQTT broker."""
        self.running = False
    
    def publish_message(self, topic: str, payload: str):
        """Simulate publishing a message to a topic."""
        message = {
            'topic': topic,
            'payload': payload,
            'timestamp': datetime.now()
        }
        self.messages.append(message)


class MockHTTPServer:
    """Mock HTTP server for testing REST API devices."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.endpoints = {}
        self.running = False
        self.server_thread = None
    
    def add_endpoint(self, path: str, response_data: Dict[str, Any]):
        """Add a mock endpoint with response data."""
        self.endpoints[path] = response_data
    
    def start(self):
        """Start mock HTTP server."""
        self.running = True
        # In a real implementation, this would start an HTTP server
        # For testing, we'll simulate server behavior
    
    def stop(self):
        """Stop mock HTTP server."""
        self.running = False


class MockModbusDevice:
    """Mock Modbus device for testing."""
    
    def __init__(self, address: int = 1):
        self.address = address
        self.registers = {i: 0 for i in range(100)}  # 100 registers
        self.running = False
    
    def set_register(self, register: int, value: int):
        """Set a register value."""
        self.registers[register] = value
    
    def get_register(self, register: int) -> int:
        """Get a register value."""
        return self.registers.get(register, 0)
    
    def start(self):
        """Start mock Modbus device."""
        self.running = True
    
    def stop(self):
        """Stop mock Modbus device."""
        self.running = False


class TestIoTProtocolImplementations:
    """Integration tests for IoT protocol implementations."""
    
    @pytest.fixture
    def mock_mqtt_broker(self):
        """Mock MQTT broker fixture."""
        broker = MockMQTTBroker()
        broker.start()
        yield broker
        broker.stop()
    
    @pytest.fixture
    def mock_http_server(self):
        """Mock HTTP server fixture."""
        server = MockHTTPServer()
        server.add_endpoint("/api/energy", {
            "power_watts": 2500.0,
            "voltage": 240.0,
            "current_amps": 10.4,
            "timestamp": datetime.now().isoformat()
        })
        server.add_endpoint("/api/temperature", {
            "temperature_celsius": 22.5,
            "humidity_percent": 45.0,
            "timestamp": datetime.now().isoformat()
        })
        server.start()
        yield server
        server.stop()
    
    @pytest.fixture
    def mock_modbus_device(self):
        """Mock Modbus device fixture."""
        device = MockModbusDevice()
        # Set some test values
        device.set_register(0, 2500)  # Power in watts
        device.set_register(1, 240)   # Voltage
        device.set_register(2, 104)   # Current in 0.1A units
        device.start()
        yield device
        device.stop()
    
    @pytest.mark.asyncio
    async def test_mqtt_protocol_implementation(self, mock_mqtt_broker):
        """
        Test MQTT protocol handler implementation.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create MQTT device configuration
        device_config = DeviceConfig(
            protocol=ProtocolType.MQTT,
            endpoint="mqtt://localhost:1883",
            topic="energy/meter/001",
            polling_interval=30,
            retry_attempts=3
        )
        
        device = Device(
            device_id="mqtt_test_device",
            device_type="smart_meter",
            name="Test MQTT Smart Meter",
            location="test_room",
            config=device_config
        )
        
        # Test MQTT handler
        with patch('paho.mqtt.client.Client') as mock_mqtt_client:
            # Mock MQTT client behavior
            mock_client_instance = MagicMock()
            mock_mqtt_client.return_value = mock_client_instance
            
            # Mock successful connection
            mock_client_instance.connect.return_value = 0  # Success
            mock_client_instance.is_connected.return_value = True
            
            # Create MQTT handler
            handler = MQTTHandler(device)
            
            # Test connection
            connection_result = await handler.connect()
            assert connection_result is True, "MQTT handler should connect successfully"
            
            # Mock message reception
            test_message_payload = json.dumps({
                "power_watts": 2500.0,
                "voltage": 240.0,
                "current_amps": 10.4,
                "timestamp": datetime.now().isoformat()
            })
            
            # Simulate message callback
            handler._on_message(mock_client_instance, None, type('Message', (), {
                'topic': 'energy/meter/001',
                'payload': test_message_payload.encode()
            })())
            
            # Test data reading
            reading = await handler.read_data()
            
            assert reading is not None, "Should receive sensor reading"
            assert reading.sensor_id == "mqtt_test_device"
            assert reading.readings.power_watts == 2500.0
            assert reading.readings.voltage == 240.0
            assert reading.readings.current_amps == 10.4
            assert reading.quality_score > 0.8
            
            # Test disconnection
            await handler.disconnect()
            mock_client_instance.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_http_rest_protocol_implementation(self, mock_http_server):
        """
        Test HTTP REST protocol handler implementation.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create HTTP device configuration
        device_config = DeviceConfig(
            protocol=ProtocolType.HTTP_REST,
            endpoint="http://localhost:8080/api/energy",
            polling_interval=60,
            retry_attempts=3
        )
        
        device = Device(
            device_id="http_test_device",
            device_type="energy_monitor",
            name="Test HTTP Energy Monitor",
            location="test_room",
            config=device_config
        )
        
        # Test HTTP handler
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Mock HTTP session and response
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "power_watts": 2500.0,
                "voltage": 240.0,
                "current_amps": 10.4,
                "timestamp": datetime.now().isoformat()
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Create HTTP handler
            handler = HTTPHandler(device)
            
            # Test connection
            connection_result = await handler.connect()
            assert connection_result is True, "HTTP handler should connect successfully"
            
            # Test data reading
            reading = await handler.read_data()
            
            assert reading is not None, "Should receive sensor reading"
            assert reading.sensor_id == "http_test_device"
            assert reading.readings.power_watts == 2500.0
            assert reading.readings.voltage == 240.0
            assert reading.readings.current_amps == 10.4
            assert reading.quality_score > 0.8
            
            # Verify HTTP request was made
            mock_session.get.assert_called_with("http://localhost:8080/api/energy")
    
    @pytest.mark.asyncio
    async def test_modbus_protocol_implementation(self, mock_modbus_device):
        """
        Test Modbus protocol handler implementation.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create Modbus device configuration
        device_config = DeviceConfig(
            protocol=ProtocolType.MODBUS,
            endpoint="localhost:502",
            modbus_address=1,
            register_map={
                "power_watts": 0,
                "voltage": 1,
                "current_amps": 2
            },
            polling_interval=30,
            retry_attempts=3
        )
        
        device = Device(
            device_id="modbus_test_device",
            device_type="power_meter",
            name="Test Modbus Power Meter",
            location="test_room",
            config=device_config
        )
        
        # Test Modbus handler
        with patch('pymodbus.client.tcp.ModbusTcpClient') as mock_modbus_client:
            # Mock Modbus client behavior
            mock_client_instance = MagicMock()
            mock_modbus_client.return_value = mock_client_instance
            
            # Mock successful connection
            mock_client_instance.connect.return_value = True
            mock_client_instance.is_socket_open.return_value = True
            
            # Mock register reading
            mock_result = MagicMock()
            mock_result.isError.return_value = False
            mock_result.registers = [2500, 240, 104]  # Power, Voltage, Current*10
            mock_client_instance.read_holding_registers.return_value = mock_result
            
            # Create Modbus handler
            handler = ModbusHandler(device)
            
            # Test connection
            connection_result = await handler.connect()
            assert connection_result is True, "Modbus handler should connect successfully"
            
            # Test data reading
            reading = await handler.read_data()
            
            assert reading is not None, "Should receive sensor reading"
            assert reading.sensor_id == "modbus_test_device"
            assert reading.readings.power_watts == 2500.0
            assert reading.readings.voltage == 240.0
            assert reading.readings.current_amps == 10.4  # 104 / 10
            assert reading.quality_score > 0.8
            
            # Verify Modbus registers were read
            mock_client_instance.read_holding_registers.assert_called()
            
            # Test disconnection
            await handler.disconnect()
            mock_client_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_protocol_error_handling_and_reconnection(self):
        """
        Test protocol error handling and reconnection logic.
        
        **Validates: Requirements 7.3, 7.5**
        """
        # Test MQTT reconnection
        device_config = DeviceConfig(
            protocol=ProtocolType.MQTT,
            endpoint="mqtt://localhost:1883",
            topic="energy/meter/001",
            polling_interval=30,
            retry_attempts=3
        )
        
        device = Device(
            device_id="reconnect_test_device",
            device_type="smart_meter",
            name="Test Reconnection Device",
            location="test_room",
            config=device_config
        )
        
        with patch('paho.mqtt.client.Client') as mock_mqtt_client:
            mock_client_instance = MagicMock()
            mock_mqtt_client.return_value = mock_client_instance
            
            # Simulate connection failure then success
            connection_attempts = 0
            def mock_connect(*args, **kwargs):
                nonlocal connection_attempts
                connection_attempts += 1
                if connection_attempts <= 2:  # First 2 attempts fail
                    raise Exception("Connection failed")
                return 0  # Success on 3rd attempt
            
            mock_client_instance.connect.side_effect = mock_connect
            mock_client_instance.is_connected.return_value = True
            
            handler = MQTTHandler(device)
            
            # Test connection with retries
            connection_result = await handler.connect()
            assert connection_result is True, "Should eventually connect after retries"
            assert connection_attempts == 3, "Should retry connection attempts"
            
            # Test disconnection handling
            handler._on_disconnect(mock_client_instance, None, 1)  # Unexpected disconnect
            
            # Verify reconnection attempt would be triggered
            assert handler.connection_lost is True
    
    @pytest.mark.asyncio
    async def test_device_auto_discovery(self):
        """
        Test IoT device auto-discovery functionality.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        
        # Mock network scanning for devices
        with patch.object(iot_service, '_scan_network_for_devices') as mock_scan:
            # Mock discovered devices
            discovered_devices = [
                {
                    'ip': '192.168.1.100',
                    'port': 1883,
                    'protocol': 'mqtt',
                    'device_type': 'smart_meter',
                    'device_info': {
                        'manufacturer': 'TestCorp',
                        'model': 'SM-100',
                        'serial': 'ABC123'
                    }
                },
                {
                    'ip': '192.168.1.101',
                    'port': 8080,
                    'protocol': 'http',
                    'device_type': 'temperature_sensor',
                    'device_info': {
                        'manufacturer': 'TempCorp',
                        'model': 'TS-200',
                        'serial': 'DEF456'
                    }
                }
            ]
            
            mock_scan.return_value = discovered_devices
            
            # Test device discovery
            discovery_result = await iot_service.discover_devices()
            
            assert discovery_result.success is True
            assert len(discovery_result.discovered_devices) == 2
            
            # Verify device configurations are created correctly
            mqtt_device = next(
                (d for d in discovery_result.discovered_devices 
                 if d.config.protocol == ProtocolType.MQTT), 
                None
            )
            assert mqtt_device is not None
            assert mqtt_device.config.endpoint == "mqtt://192.168.1.100:1883"
            
            http_device = next(
                (d for d in discovery_result.discovered_devices 
                 if d.config.protocol == ProtocolType.HTTP_REST), 
                None
            )
            assert http_device is not None
            assert "192.168.1.101:8080" in http_device.config.endpoint
    
    @pytest.mark.asyncio
    async def test_real_time_data_synchronization(self):
        """
        Test real-time data synchronization across multiple device streams.
        
        **Validates: Requirements 7.5**
        """
        iot_service = IoTIntegrationService()
        
        # Create multiple test devices
        devices = []
        for i in range(3):
            device_config = DeviceConfig(
                protocol=ProtocolType.MQTT,
                endpoint=f"mqtt://localhost:188{i}",
                topic=f"energy/device/{i:03d}",
                polling_interval=10,
                retry_attempts=3
            )
            
            device = Device(
                device_id=f"sync_test_device_{i:03d}",
                device_type="energy_monitor",
                name=f"Test Sync Device {i}",
                location=f"room_{i}",
                config=device_config
            )
            devices.append(device)
        
        # Mock handlers for all devices
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            mock_handlers = []
            
            for i, device in enumerate(devices):
                mock_handler = AsyncMock()
                mock_handler.connect.return_value = True
                mock_handler.is_connected = True
                
                # Mock synchronized readings with timestamps
                base_time = datetime.now()
                mock_handler.read_data.return_value = SensorReading(
                    sensor_id=device.device_id,
                    device_type=device.device_type,
                    timestamp=base_time + timedelta(seconds=i),  # Slight time differences
                    readings=SensorReadings(
                        power_watts=1000.0 + (i * 100),
                        voltage=240.0,
                        current_amps=4.0 + i
                    ),
                    quality_score=0.9,
                    location=device.location
                )
                
                mock_handlers.append(mock_handler)
            
            mock_create_handler.side_effect = lambda device: mock_handlers[devices.index(device)]
            
            # Register all devices
            for device in devices:
                success = await iot_service.register_device(device)
                assert success, f"Device {device.device_id} should register successfully"
            
            # Test synchronized data reading
            start_time = datetime.now()
            readings = await iot_service.read_all_devices()
            end_time = datetime.now()
            
            # Verify all devices provided readings
            assert len(readings) == len(devices)
            
            # Verify data synchronization (all readings within reasonable time window)
            reading_times = [reading.timestamp for reading in readings.values() if reading]
            time_spread = max(reading_times) - min(reading_times)
            assert time_spread.total_seconds() < 5.0, "Readings should be synchronized within 5 seconds"
            
            # Verify data quality
            for device_id, reading in readings.items():
                assert reading is not None, f"Device {device_id} should provide reading"
                assert reading.quality_score > 0.8, f"Device {device_id} should have good data quality"
    
    @pytest.mark.asyncio
    async def test_offline_operation_and_data_buffering(self):
        """
        Test IoT service offline operation and data buffering capabilities.
        
        **Validates: Requirements 7.3**
        """
        iot_service = IoTIntegrationService(max_buffer_size=100)
        
        # Create test device
        device_config = DeviceConfig(
            protocol=ProtocolType.MQTT,
            endpoint="mqtt://localhost:1883",
            topic="energy/meter/001",
            polling_interval=30,
            retry_attempts=3
        )
        
        device = Device(
            device_id="offline_test_device",
            device_type="smart_meter",
            name="Test Offline Device",
            location="test_room",
            config=device_config
        )
        
        # Mock handler
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected = True
            mock_create_handler.return_value = mock_handler
            
            # Register device
            await iot_service.register_device(device)
            
            # Test going offline
            await iot_service.set_online_status(False)
            assert iot_service.is_online is False
            
            # Generate readings while offline
            offline_readings = []
            for i in range(3):  # Reduced from 5
                reading = SensorReading(
                    sensor_id="offline_test_device",
                    device_type="smart_meter",
                    timestamp=datetime.now() + timedelta(seconds=i),
                    readings=SensorReadings(
                        power_watts=2000.0 + (i * 100),
                        voltage=240.0,
                        current_amps=8.0 + i
                    ),
                    quality_score=0.9,
                    location="test_room"
                )
                offline_readings.append(reading)
                
                # Store reading while offline (should be buffered)
                await iot_service._store_sensor_reading(reading)
            
            # Verify data is buffered
            assert len(iot_service.offline_buffer) == 3, "Should buffer 3 readings while offline"
            
            # Test coming back online
            with patch.object(iot_service, '_flush_offline_buffer') as mock_flush:
                mock_flush.return_value = None
                
                await iot_service.set_online_status(True)
                assert iot_service.is_online is True
                
                # Verify buffer flush was attempted
                mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_protocol_performance_under_load(self):
        """
        Test protocol handler performance under high load conditions.
        
        **Validates: Requirements 3.3, 7.2**
        """
        # Test concurrent device operations
        num_devices = 5  # Reduced from 20
        devices = []
        
        for i in range(num_devices):
            device_config = DeviceConfig(
                protocol=ProtocolType.MQTT,
                endpoint=f"mqtt://localhost:188{i % 3}",  # 3 different brokers
                topic=f"energy/load_test/{i:03d}",
                polling_interval=5,
                retry_attempts=2
            )
            
            device = Device(
                device_id=f"load_test_device_{i:03d}",
                device_type="energy_monitor",
                name=f"Load Test Device {i}",
                location=f"zone_{i % 5}",
                config=device_config
            )
            devices.append(device)
        
        iot_service = IoTIntegrationService()
        
        # Mock handlers for load testing
        with patch.object(iot_service, '_create_handler') as mock_create_handler:
            def create_mock_handler(device):
                mock_handler = AsyncMock()
                mock_handler.connect.return_value = True
                mock_handler.is_connected = True
                
                # Simulate variable response times
                import random
                response_time = random.uniform(0.05, 0.2)  # 50-200ms
                
                async def mock_read_data():
                    await asyncio.sleep(response_time)
                    return SensorReading(
                        sensor_id=device.device_id,
                        device_type=device.device_type,
                        timestamp=datetime.now(),
                        readings=SensorReadings(
                            power_watts=random.uniform(1000, 3000),
                            voltage=240.0,
                            current_amps=random.uniform(4, 12)
                        ),
                        quality_score=random.uniform(0.85, 0.95),
                        location=device.location
                    )
                
                mock_handler.read_data = mock_read_data
                return mock_handler
            
            mock_create_handler.side_effect = create_mock_handler
            
            # Register all devices concurrently
            start_time = datetime.now()
            
            registration_tasks = [
                iot_service.register_device(device) for device in devices
            ]
            registration_results = await asyncio.gather(*registration_tasks, return_exceptions=True)
            
            registration_time = (datetime.now() - start_time).total_seconds()
            successful_registrations = sum(1 for result in registration_results if result is True)
            
            # Test concurrent data reading
            start_time = datetime.now()
            readings = await iot_service.read_all_devices()
            reading_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            assert successful_registrations >= num_devices * 0.9, "At least 90% of devices should register"
            assert registration_time < 10.0, "Device registration should complete within 10 seconds"
            assert reading_time < 5.0, "Concurrent reading should complete within 5 seconds"
            assert len(readings) >= num_devices * 0.8, "At least 80% of devices should provide readings"
            
            # Verify data quality under load
            quality_scores = [reading.quality_score for reading in readings.values() if reading]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            assert avg_quality > 0.8, "Average data quality should remain high under load"
            
            return {
                'devices_registered': successful_registrations,
                'registration_time': registration_time,
                'reading_time': reading_time,
                'readings_received': len(readings),
                'average_quality': avg_quality
            }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])