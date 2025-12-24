"""
Property-Based Tests for IoT Protocol Support

**Feature: smart-energy-copilot, Property 7: IoT protocol compatibility**
**Validates: Requirements 7.1**

Tests that the Smart Energy Copilot supports standard IoT protocols including MQTT, HTTP REST APIs, and Modbus.
"""

import pytest
from hypothesis import given, strategies as st, settings
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.models.device import Device, DeviceConfig, ProtocolType
from src.services.iot_protocols.mqtt_handler import MQTTHandler
from src.services.iot_protocols.http_handler import HTTPHandler
from src.services.iot_protocols.modbus_handler import ModbusHandler
from src.services.iot_integration import IoTIntegrationService


# Strategy for generating valid device configurations
@st.composite
def device_config_strategy(draw, protocol: ProtocolType):
    """Generate valid device configurations for different protocols"""
    if protocol == ProtocolType.MQTT:
        endpoint = draw(st.sampled_from([
            "mqtt://localhost:1883",
            "mqtt://192.168.1.100:1883",
            "mqtts://broker.example.com:8883"
        ]))
        credentials = draw(st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "username": st.text(min_size=1, max_size=20),
                "password": st.text(min_size=1, max_size=20)
            })
        ))
    elif protocol == ProtocolType.HTTP_REST:
        endpoint = draw(st.sampled_from([
            "http://localhost:8080",
            "http://192.168.1.100:3000",
            "https://api.example.com"
        ]))
        credentials = draw(st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "auth_type": st.sampled_from(["basic", "bearer", "api_key"]),
                "username": st.text(min_size=1, max_size=20),
                "password": st.text(min_size=1, max_size=20)
            }),
            st.fixed_dictionaries({
                "auth_type": st.just("bearer"),
                "token": st.text(min_size=10, max_size=50)
            }),
            st.fixed_dictionaries({
                "auth_type": st.just("api_key"),
                "api_key": st.text(min_size=10, max_size=50),
                "key_header": st.sampled_from(["X-API-Key", "Authorization"])
            })
        ))
    elif protocol == ProtocolType.MODBUS:
        endpoint = draw(st.sampled_from([
            "modbus+tcp://localhost:502",
            "modbus+tcp://192.168.1.100:502",
            "modbus+rtu:///dev/ttyUSB0?baudrate=9600"
        ]))
        credentials = draw(st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "slave_id": st.integers(min_value=1, max_value=247)
            })
        ))
    else:
        endpoint = "unknown://localhost"
        credentials = None
    
    return DeviceConfig(
        protocol=protocol,
        endpoint=endpoint,
        credentials=credentials,
        polling_interval=draw(st.integers(min_value=1, max_value=300)),
        timeout=draw(st.integers(min_value=5, max_value=60)),
        retry_attempts=draw(st.integers(min_value=0, max_value=5))
    )


@st.composite
def device_strategy(draw, protocol: ProtocolType):
    """Generate valid devices for different protocols"""
    config = draw(device_config_strategy(protocol))
    
    return Device(
        device_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")).filter(lambda x: x.strip())),
        device_type=draw(st.sampled_from(["smart_meter", "temperature_sensor", "power_monitor", "occupancy_sensor"])),
        name=draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())),
        location=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        config=config,
        metadata=draw(st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                min_size=0,
                max_size=5
            )
        ))
    )


class TestIoTProtocolSupport:
    """Property-based tests for IoT protocol support"""

    @given(protocol=st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]))
    @settings(max_examples=5, deadline=5000)
    def test_protocol_handler_creation(self, protocol: ProtocolType):
        """
        Property 7: IoT protocol compatibility
        
        For any supported protocol type, the system should be able to create
        an appropriate protocol handler.
        
        **Validates: Requirements 7.1**
        """
        # Generate a device with the given protocol
        device = Device(
            device_id="test_device",
            device_type="test_type",
            name="Test Device",
            location="test_location",
            config=DeviceConfig(
                protocol=protocol,
                endpoint=f"{protocol.value}://localhost",
                polling_interval=60,
                timeout=30
            )
        )
        
        # Create IoT integration service
        iot_service = IoTIntegrationService()
        
        # Test that handler can be created for the protocol
        handler = iot_service._create_handler(device)
        
        # Verify handler is created and is of correct type
        assert handler is not None, f"Handler should be created for protocol {protocol}"
        
        if protocol == ProtocolType.MQTT:
            assert isinstance(handler, MQTTHandler), f"MQTT protocol should create MQTTHandler"
        elif protocol == ProtocolType.HTTP_REST:
            assert isinstance(handler, HTTPHandler), f"HTTP_REST protocol should create HTTPHandler"
        elif protocol == ProtocolType.MODBUS:
            assert isinstance(handler, ModbusHandler), f"MODBUS protocol should create ModbusHandler"
        
        # Verify handler has required methods
        assert hasattr(handler, 'connect'), "Handler should have connect method"
        assert hasattr(handler, 'disconnect'), "Handler should have disconnect method"
        assert hasattr(handler, 'read_data'), "Handler should have read_data method"
        assert hasattr(handler, 'ping'), "Handler should have ping method"
        assert hasattr(handler, 'discover_devices'), "Handler should have discover_devices method"

    @given(device=device_strategy(ProtocolType.MQTT))
    @settings(max_examples=5, deadline=5000)
    def test_mqtt_handler_interface_compliance(self, device: Device):
        """
        Property 7: IoT protocol compatibility - MQTT specific
        
        For any valid MQTT device configuration, the MQTT handler should
        implement the required interface methods correctly.
        
        **Validates: Requirements 7.1**
        """
        handler = MQTTHandler(device)
        
        # Test interface compliance
        assert hasattr(handler, 'device'), "Handler should store device reference"
        assert handler.device == device, "Handler should store correct device"
        assert hasattr(handler, 'is_connected'), "Handler should have is_connected property"
        assert hasattr(handler, 'last_error'), "Handler should have last_error property"
        
        # Test initial state
        assert not handler.is_connected, "Handler should start disconnected"
        assert handler.last_error is None, "Handler should start with no error"

    @given(device=device_strategy(ProtocolType.HTTP_REST))
    @settings(max_examples=5, deadline=5000)
    def test_http_handler_interface_compliance(self, device: Device):
        """
        Property 7: IoT protocol compatibility - HTTP REST specific
        
        For any valid HTTP REST device configuration, the HTTP handler should
        implement the required interface methods correctly.
        
        **Validates: Requirements 7.1**
        """
        handler = HTTPHandler(device)
        
        # Test interface compliance
        assert hasattr(handler, 'device'), "Handler should store device reference"
        assert handler.device == device, "Handler should store correct device"
        assert hasattr(handler, 'is_connected'), "Handler should have is_connected property"
        assert hasattr(handler, 'last_error'), "Handler should have last_error property"
        
        # Test initial state
        assert not handler.is_connected, "Handler should start disconnected"
        assert handler.last_error is None, "Handler should start with no error"
        
        # Test base URL parsing
        assert hasattr(handler, 'base_url'), "HTTP handler should have base_url"
        assert handler.base_url == device.config.endpoint.rstrip('/'), "Base URL should be parsed correctly"

    @given(device=device_strategy(ProtocolType.MODBUS))
    @settings(max_examples=5, deadline=5000)
    def test_modbus_handler_interface_compliance(self, device: Device):
        """
        Property 7: IoT protocol compatibility - Modbus specific
        
        For any valid Modbus device configuration, the Modbus handler should
        implement the required interface methods correctly.
        
        **Validates: Requirements 7.1**
        """
        handler = ModbusHandler(device)
        
        # Test interface compliance
        assert hasattr(handler, 'device'), "Handler should store device reference"
        assert handler.device == device, "Handler should store correct device"
        assert hasattr(handler, 'is_connected'), "Handler should have is_connected property"
        assert hasattr(handler, 'last_error'), "Handler should have last_error property"
        
        # Test initial state
        assert not handler.is_connected, "Handler should start disconnected"
        assert handler.last_error is None, "Handler should start with no error"
        
        # Test Modbus-specific attributes
        assert hasattr(handler, 'slave_id'), "Modbus handler should have slave_id"
        assert isinstance(handler.slave_id, int), "Slave ID should be integer"
        assert 1 <= handler.slave_id <= 247, "Slave ID should be in valid range"

    @given(
        devices=st.lists(
            st.one_of(
                device_strategy(ProtocolType.MQTT),
                device_strategy(ProtocolType.HTTP_REST),
                device_strategy(ProtocolType.MODBUS)
            ),
            min_size=1,
            max_size=10,
            unique_by=lambda d: d.device_id
        )
    )
    @settings(max_examples=5, deadline=10000)
    def test_multiple_protocol_support(self, devices):
        """
        Property 7: IoT protocol compatibility - Multiple protocols
        
        For any collection of devices with different protocols, the system
        should be able to create handlers for all supported protocols.
        
        **Validates: Requirements 7.1**
        """
        iot_service = IoTIntegrationService()
        
        # Test that handlers can be created for all devices
        handlers = {}
        for device in devices:
            handler = iot_service._create_handler(device)
            assert handler is not None, f"Handler should be created for device {device.device_id} with protocol {device.config.protocol}"
            handlers[device.device_id] = handler
        
        # Verify all handlers are of correct types
        for device in devices:
            handler = handlers[device.device_id]
            protocol = device.config.protocol
            
            if protocol == ProtocolType.MQTT:
                assert isinstance(handler, MQTTHandler)
            elif protocol == ProtocolType.HTTP_REST:
                assert isinstance(handler, HTTPHandler)
            elif protocol == ProtocolType.MODBUS:
                assert isinstance(handler, ModbusHandler)
        
        # Verify handlers are independent
        assert len(set(id(h) for h in handlers.values())) == len(handlers), "Each device should have its own handler instance"

    @given(protocol=st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]))
    @settings(max_examples=5, deadline=5000)
    def test_protocol_handler_error_handling(self, protocol: ProtocolType):
        """
        Property 7: IoT protocol compatibility - Error handling
        
        For any protocol handler, error conditions should be handled gracefully
        without crashing the system.
        
        **Validates: Requirements 7.1**
        """
        # Create device with potentially problematic configuration
        device = Device(
            device_id="error_test_device",
            device_type="test_type",
            name="Error Test Device",
            location="test_location",
            config=DeviceConfig(
                protocol=protocol,
                endpoint="invalid://nonexistent.host:99999",  # Invalid endpoint
                polling_interval=1,
                timeout=1  # Very short timeout
            )
        )
        
        iot_service = IoTIntegrationService()
        handler = iot_service._create_handler(device)
        
        assert handler is not None, "Handler should be created even with invalid config"
        
        # Test that error methods don't crash
        assert handler.last_error is None, "Handler should start with no error"
        assert not handler.is_connected, "Handler should start disconnected"
        
        # Test error state management
        handler._set_error("Test error")
        assert handler.last_error == "Test error", "Error should be stored"
        assert not handler.is_connected, "Handler should be disconnected on error"
        
        handler._clear_error()
        assert handler.last_error is None, "Error should be cleared"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
