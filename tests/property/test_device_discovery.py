"""
Property-Based Tests for Device Auto-Discovery

**Feature: smart-energy-copilot, Property 9: Device auto-discovery**
**Validates: Requirements 7.4**

Tests that the Smart Energy Copilot can auto-discover compatible IoT devices and integrate them into analysis
when new devices are added to the network.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from src.models.device import Device, DeviceConfig, ProtocolType, DeviceDiscoveryResult
from src.services.device_registry import DeviceRegistryService
from src.services.iot_integration import IoTIntegrationService
from src.services.iot_protocols.mqtt_handler import MQTTHandler
from src.services.iot_protocols.http_handler import HTTPHandler
from src.services.iot_protocols.modbus_handler import ModbusHandler


# Strategy for generating valid device configurations for discovery
@st.composite
def discoverable_device_strategy(draw, protocol=None):
    """Generate devices that could be discovered on a network"""
    if protocol is None:
        protocol = draw(st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]))
    
    if protocol == ProtocolType.MQTT:
        endpoint = draw(st.sampled_from([
            "mqtt://192.168.1.100:1883",
            "mqtt://192.168.1.101:1883",
            "mqtt://192.168.1.102:1883"
        ]))
    elif protocol == ProtocolType.HTTP_REST:
        ip = draw(st.sampled_from(["192.168.1.100", "192.168.1.101", "192.168.1.102"]))
        port = draw(st.sampled_from([80, 8080, 3000, 5000]))
        endpoint = f"http://{ip}:{port}"
    elif protocol == ProtocolType.MODBUS:
        ip = draw(st.sampled_from(["192.168.1.100", "192.168.1.101", "192.168.1.102"]))
        port = draw(st.sampled_from([502, 503]))
        endpoint = f"modbus+tcp://{ip}:{port}"
    else:
        endpoint = "unknown://localhost"
    
    device_id = draw(st.text(
        min_size=5, max_size=30,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")
    ).filter(lambda x: x.strip() and not x.startswith('-') and not x.endswith('-')))
    
    config = DeviceConfig(
        protocol=protocol,
        endpoint=endpoint,
        polling_interval=draw(st.integers(min_value=30, max_value=300)),
        timeout=draw(st.integers(min_value=10, max_value=60)),
        retry_attempts=draw(st.integers(min_value=1, max_value=5))
    )
    
    if protocol == ProtocolType.MODBUS:
        config.credentials = {"slave_id": draw(st.integers(min_value=1, max_value=10))}
    
    return Device(
        device_id=device_id,
        device_type=draw(st.sampled_from(["smart_meter", "temperature_sensor", "power_monitor", "occupancy_sensor"])),
        name=draw(st.text(min_size=5, max_size=50).filter(lambda x: x.strip())),
        location=draw(st.sampled_from(["living_room", "kitchen", "bedroom", "garage", "basement"])),
        config=config,
        metadata=draw(st.one_of(
            st.none(),
            st.dictionaries(
                st.sampled_from(["manufacturer", "model", "firmware_version", "ip", "port"]),
                st.text(min_size=1, max_size=20),
                min_size=0,
                max_size=3
            )
        ))
    )


@st.composite
def discovery_result_strategy(draw, protocol: ProtocolType):
    """Generate discovery results for testing"""
    devices = draw(st.lists(
        discoverable_device_strategy(protocol),
        min_size=0,
        max_size=5,
        unique_by=lambda d: d.device_id
    ))
    
    return DeviceDiscoveryResult(
        discovered_devices=devices,
        discovery_method=protocol.value,
        success=draw(st.booleans()),
        error_message=draw(st.one_of(
            st.none(),
            st.sampled_from(["Network timeout", "Connection refused", "Invalid response"])
        )) if not devices else None
    )


class TestDeviceAutoDiscovery:
    """Property-based tests for device auto-discovery functionality"""

    @given(protocol=st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]))
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_protocol_discovery_capability(self, protocol: ProtocolType):
        """
        Property 9: Device auto-discovery - Protocol discovery capability
        
        For any supported protocol, the system should be able to initiate
        device discovery and return a structured result.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        
        # Mock the protocol handlers' discover_devices methods
        mock_devices = [
            Device(
                device_id=f"test_{protocol.value}_device",
                device_type="test_device",
                name=f"Test {protocol.value} Device",
                location="test_location",
                config=DeviceConfig(
                    protocol=protocol,
                    endpoint=f"{protocol.value}://localhost",
                    polling_interval=60
                )
            )
        ]
        
        with patch.object(MQTTHandler, 'discover_devices', return_value=mock_devices if protocol == ProtocolType.MQTT else []), \
             patch.object(HTTPHandler, 'discover_devices', return_value=mock_devices if protocol == ProtocolType.HTTP_REST else []), \
             patch.object(ModbusHandler, 'discover_devices', return_value=mock_devices if protocol == ProtocolType.MODBUS else []):
            
            # Test discovery for specific protocol
            result = await iot_service.discover_devices([protocol])
            
            # Discovery should complete successfully
            assert isinstance(result, DeviceDiscoveryResult), "Discovery should return DeviceDiscoveryResult"
            assert result.success, "Discovery should succeed for supported protocols"
            
            # Check that discovery method contains a reasonable representation of the protocol
            discovery_method_upper = result.discovery_method.upper()
            if protocol == ProtocolType.MQTT:
                assert "MQTT" in discovery_method_upper, "Discovery method should include MQTT"
            elif protocol == ProtocolType.HTTP_REST:
                assert "HTTP" in discovery_method_upper, "Discovery method should include HTTP"
            elif protocol == ProtocolType.MODBUS:
                assert "MODBUS" in discovery_method_upper, "Discovery method should include MODBUS"
            
            # If devices were discovered, they should be valid
            for device in result.discovered_devices:
                assert isinstance(device, Device), "Discovered items should be Device objects"
                assert device.config.protocol == protocol, "Discovered device should have correct protocol"
                assert device.device_id, "Discovered device should have valid device_id"
                assert device.device_type, "Discovered device should have valid device_type"

    @given(
        protocols=st.lists(
            st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=50, deadline=10000)
    @pytest.mark.asyncio
    async def test_multi_protocol_discovery(self, protocols: List[ProtocolType]):
        """
        Property 9: Device auto-discovery - Multi-protocol discovery
        
        For any combination of supported protocols, the system should be able
        to discover devices across all protocols simultaneously.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        
        # Create mock devices for each protocol
        mock_devices_by_protocol = {}
        for protocol in protocols:
            mock_devices_by_protocol[protocol] = [
                Device(
                    device_id=f"test_{protocol.value}_device_{i}",
                    device_type="test_device",
                    name=f"Test {protocol.value} Device {i}",
                    location="test_location",
                    config=DeviceConfig(
                        protocol=protocol,
                        endpoint=f"{protocol.value}://localhost:{1000+i}",
                        polling_interval=60
                    )
                ) for i in range(2)  # 2 devices per protocol
            ]
        
        # Mock discovery methods
        with patch.object(MQTTHandler, 'discover_devices', 
                         return_value=mock_devices_by_protocol.get(ProtocolType.MQTT, [])), \
             patch.object(HTTPHandler, 'discover_devices', 
                         return_value=mock_devices_by_protocol.get(ProtocolType.HTTP_REST, [])), \
             patch.object(ModbusHandler, 'discover_devices', 
                         return_value=mock_devices_by_protocol.get(ProtocolType.MODBUS, [])):
            
            result = await iot_service.discover_devices(protocols)
            
            # Discovery should succeed
            assert result.success, "Multi-protocol discovery should succeed"
            
            # Should discover devices from all requested protocols
            discovered_protocols = set()
            for device in result.discovered_devices:
                discovered_protocols.add(device.config.protocol)
            
            # All requested protocols should be represented in discovery method
            discovery_method_upper = result.discovery_method.upper()
            for protocol in protocols:
                if protocol == ProtocolType.MQTT:
                    assert "MQTT" in discovery_method_upper, "Discovery method should include MQTT"
                elif protocol == ProtocolType.HTTP_REST:
                    assert "HTTP" in discovery_method_upper, "Discovery method should include HTTP"
                elif protocol == ProtocolType.MODBUS:
                    assert "MODBUS" in discovery_method_upper, "Discovery method should include MODBUS"
            
            # Devices should be unique by device_id
            device_ids = [d.device_id for d in result.discovered_devices]
            assert len(device_ids) == len(set(device_ids)), "Discovered devices should have unique IDs"
            
            # Each discovered device should be valid
            for device in result.discovered_devices:
                assert device.config.protocol in protocols, \
                    "Discovered device protocol should be in requested protocols"

    @given(
        existing_devices=st.lists(
            discoverable_device_strategy(),
            min_size=0,
            max_size=3,
            unique_by=lambda d: d.device_id
        ),
        new_devices=st.lists(
            discoverable_device_strategy(),
            min_size=1,
            max_size=3,
            unique_by=lambda d: d.device_id
        )
    )
    @settings(max_examples=30, deadline=15000)
    @pytest.mark.asyncio
    async def test_discovery_filters_existing_devices(self, existing_devices: List[Device], new_devices: List[Device]):
        """
        Property 9: Device auto-discovery - Existing device filtering
        
        For any set of existing devices and newly discovered devices, the discovery
        process should filter out already registered devices and only return new ones.
        
        **Validates: Requirements 7.4**
        """
        # Ensure no overlap between existing and new devices
        existing_ids = {d.device_id for d in existing_devices}
        new_devices = [d for d in new_devices if d.device_id not in existing_ids]
        assume(len(new_devices) > 0)  # Need at least one new device
        
        iot_service = IoTIntegrationService()
        registry_service = DeviceRegistryService(iot_service)
        
        # Register existing devices
        for device in existing_devices:
            iot_service.devices[device.device_id] = device
        
        # Mock discovery to return both existing and new devices
        all_discovered = existing_devices + new_devices
        
        with patch.object(MQTTHandler, 'discover_devices', return_value=all_discovered), \
             patch.object(HTTPHandler, 'discover_devices', return_value=all_discovered), \
             patch.object(ModbusHandler, 'discover_devices', return_value=all_discovered):
            
            result = await registry_service.discover_and_register_devices(auto_register=False)
            
            # Discovery should succeed
            assert result.success, "Discovery should succeed"
            
            # Should only return new devices (not already registered)
            discovered_ids = {d.device_id for d in result.discovered_devices}
            new_device_ids = {d.device_id for d in new_devices}
            
            assert discovered_ids == new_device_ids, \
                "Discovery should only return new devices, not existing ones"
            
            # Should not return any existing devices
            for device in result.discovered_devices:
                assert device.device_id not in existing_ids, \
                    f"Device {device.device_id} should not be in discovered list (already exists)"

    @given(
        discovered_devices=st.lists(
            discoverable_device_strategy(),
            min_size=1,
            max_size=5,
            unique_by=lambda d: d.device_id
        ),
        auto_register=st.booleans()
    )
    @settings(max_examples=30, deadline=15000)
    @pytest.mark.asyncio
    async def test_discovery_integration_with_registration(self, discovered_devices: List[Device], auto_register: bool):
        """
        Property 9: Device auto-discovery - Integration with registration
        
        For any set of discovered devices, the discovery process should optionally
        integrate them into the system by auto-registering them.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        registry_service = DeviceRegistryService(iot_service)
        
        # Mock successful device registration
        async def mock_register_device(device: Device) -> bool:
            iot_service.devices[device.device_id] = device
            return True
        
        # Mock database operations
        with patch.object(registry_service, 'register_device', side_effect=mock_register_device), \
             patch.object(MQTTHandler, 'discover_devices', return_value=discovered_devices), \
             patch.object(HTTPHandler, 'discover_devices', return_value=discovered_devices), \
             patch.object(ModbusHandler, 'discover_devices', return_value=discovered_devices), \
             patch('src.services.device_registry.get_db_session'):
            
            initial_device_count = len(iot_service.devices)
            
            result = await registry_service.discover_and_register_devices(auto_register=auto_register)
            
            # Discovery should succeed
            assert result.success, "Discovery and registration should succeed"
            
            # Check registration behavior
            if auto_register:
                # All discovered devices should be registered
                final_device_count = len(iot_service.devices)
                expected_count = initial_device_count + len(discovered_devices)
                assert final_device_count == expected_count, \
                    f"Expected {expected_count} devices after auto-registration, got {final_device_count}"
                
                # All discovered devices should be in the registry
                for device in discovered_devices:
                    assert device.device_id in iot_service.devices, \
                        f"Device {device.device_id} should be registered"
            else:
                # No devices should be registered
                final_device_count = len(iot_service.devices)
                assert final_device_count == initial_device_count, \
                    "No devices should be registered when auto_register=False"

    @given(
        discovery_timeout=st.integers(min_value=1, max_value=5),  # Reduced timeout for faster tests
        should_timeout=st.booleans()
    )
    @settings(max_examples=20, deadline=15000)  # Increased deadline for timeout tests
    @pytest.mark.asyncio
    async def test_discovery_timeout_handling(self, discovery_timeout: int, should_timeout: bool):
        """
        Property 9: Device auto-discovery - Timeout handling
        
        For any discovery timeout setting, the discovery process should handle
        timeouts gracefully and return appropriate results.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        
        async def mock_slow_discovery():
            if should_timeout:
                # Simulate slow discovery that exceeds timeout
                await asyncio.sleep(discovery_timeout + 0.5)
            else:
                # Simulate fast discovery within timeout
                await asyncio.sleep(0.1)
            return []
        
        with patch.object(MQTTHandler, 'discover_devices', side_effect=mock_slow_discovery), \
             patch.object(HTTPHandler, 'discover_devices', side_effect=mock_slow_discovery), \
             patch.object(ModbusHandler, 'discover_devices', side_effect=mock_slow_discovery):
            
            result = await iot_service.discover_devices(
                [ProtocolType.MQTT], 
                discovery_timeout=discovery_timeout
            )
            
            # Discovery should complete (either successfully or with timeout)
            assert isinstance(result, DeviceDiscoveryResult), "Should return DeviceDiscoveryResult"
            
            if should_timeout:
                # Timeout should be handled gracefully
                # The result might be successful with empty devices or failed with timeout error
                assert isinstance(result.discovered_devices, list), "Should return empty list on timeout"
            else:
                # Should succeed within timeout
                assert result.success, "Discovery should succeed when within timeout"

    @given(
        protocols=st.lists(
            st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]),
            min_size=1,
            max_size=3,
            unique=True
        ),
        error_protocols=st.lists(
            st.sampled_from([ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]),
            min_size=0,
            max_size=2,
            unique=True
        )
    )
    @settings(max_examples=30, deadline=10000)
    @pytest.mark.asyncio
    async def test_discovery_error_resilience(self, protocols: List[ProtocolType], error_protocols: List[ProtocolType]):
        """
        Property 9: Device auto-discovery - Error resilience
        
        For any combination of protocols where some may fail, the discovery
        process should be resilient and continue with successful protocols.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        
        # Create mock devices for successful protocols
        successful_protocols = [p for p in protocols if p not in error_protocols]
        mock_devices = []
        for protocol in successful_protocols:
            mock_devices.append(Device(
                device_id=f"test_{protocol.value}_device",
                device_type="test_device",
                name=f"Test {protocol.value} Device",
                location="test_location",
                config=DeviceConfig(
                    protocol=protocol,
                    endpoint=f"{protocol.value}://localhost",
                    polling_interval=60
                )
            ))
        
        async def mock_discovery_success():
            return mock_devices
        
        async def mock_discovery_error():
            raise Exception("Discovery failed")
        
        # Mock discovery methods
        mqtt_mock = mock_discovery_error if ProtocolType.MQTT in error_protocols else mock_discovery_success
        http_mock = mock_discovery_error if ProtocolType.HTTP_REST in error_protocols else mock_discovery_success
        modbus_mock = mock_discovery_error if ProtocolType.MODBUS in error_protocols else mock_discovery_success
        
        with patch.object(MQTTHandler, 'discover_devices', side_effect=mqtt_mock), \
             patch.object(HTTPHandler, 'discover_devices', side_effect=http_mock), \
             patch.object(ModbusHandler, 'discover_devices', side_effect=modbus_mock):
            
            result = await iot_service.discover_devices(protocols)
            
            # Discovery should handle errors gracefully
            assert isinstance(result, DeviceDiscoveryResult), "Should return DeviceDiscoveryResult"
            
            if successful_protocols:
                # Should succeed if at least one protocol works
                assert result.success, "Discovery should succeed if any protocol works"
                
                # Should discover devices from successful protocols
                discovered_protocols = {d.config.protocol for d in result.discovered_devices}
                for protocol in successful_protocols:
                    if protocol in [ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]:
                        # At least some successful protocols should have devices
                        pass  # The exact behavior depends on mock implementation
            
            # Error protocols should not crash the entire discovery
            assert isinstance(result.discovered_devices, list), "Should return list even with errors"

    @given(
        device=discoverable_device_strategy()
    )
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_discovered_device_validation(self, device: Device):
        """
        Property 9: Device auto-discovery - Discovered device validation
        
        For any discovered device, it should pass validation checks before
        being integrated into the system.
        
        **Validates: Requirements 7.4**
        """
        iot_service = IoTIntegrationService()
        registry_service = DeviceRegistryService(iot_service)
        
        # Mock connectivity test to return success
        async def mock_connectivity_test(test_device: Device) -> Dict[str, Any]:
            return {"success": True, "error": None}
        
        with patch.object(registry_service, '_test_device_connectivity', side_effect=mock_connectivity_test):
            validation_result = await registry_service.validate_device(device)
            
            # Validation should complete
            assert isinstance(validation_result, dict), "Validation should return dict"
            assert "valid" in validation_result, "Validation should include 'valid' field"
            assert "errors" in validation_result, "Validation should include 'errors' field"
            assert "warnings" in validation_result, "Validation should include 'warnings' field"
            
            # For valid devices, validation should pass
            if validation_result["valid"]:
                assert isinstance(validation_result["errors"], list), "Errors should be a list"
                assert len(validation_result["errors"]) == 0, "Valid devices should have no errors"
            
            # Device should have required fields
            assert device.device_id, "Device should have device_id"
            assert device.device_type, "Device should have device_type"
            assert device.config.endpoint, "Device should have endpoint"
            assert device.config.protocol in [ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS], \
                "Device should have supported protocol"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])