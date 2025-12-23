"""
Device Registry Service

Manages device auto-discovery, registration, and lifecycle management.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
import logging

from src.models.device import Device, DeviceStatus, ProtocolType, DeviceDiscoveryResult
from src.services.iot_integration import IoTIntegrationService
from src.database.connection import get_db_session
from src.models.device import DeviceDB

logger = logging.getLogger(__name__)


class DeviceRegistryService:
    """Service for managing IoT device registration and discovery"""

    def __init__(self, iot_service: IoTIntegrationService):
        self.iot_service = iot_service
        self.discovery_history: List[DeviceDiscoveryResult] = []
        self.auto_discovery_enabled = True
        self.discovery_interval = 300  # 5 minutes
        self.discovery_task: Optional[asyncio.Task] = None

    async def start_auto_discovery(self):
        """Start automatic device discovery"""
        if self.discovery_task and not self.discovery_task.done():
            return  # Already running

        async def discovery_loop():
            while self.auto_discovery_enabled:
                try:
                    logger.info("Starting automatic device discovery")
                    result = await self.discover_and_register_devices()
                    
                    if result.discovered_devices:
                        logger.info(f"Auto-discovery found {len(result.discovered_devices)} new devices")
                    
                    await asyncio.sleep(self.discovery_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-discovery loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        self.discovery_task = asyncio.create_task(discovery_loop())

    async def stop_auto_discovery(self):
        """Stop automatic device discovery"""
        self.auto_discovery_enabled = False
        
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
            self.discovery_task = None

    async def discover_and_register_devices(self, 
                                          protocols: List[ProtocolType] = None,
                                          auto_register: bool = True) -> DeviceDiscoveryResult:
        """Discover devices and optionally auto-register them"""
        try:
            # Perform discovery
            result = await self.iot_service.discover_devices(protocols)
            
            # Store discovery result
            self.discovery_history.append(result)
            
            # Keep only last 100 discovery results
            if len(self.discovery_history) > 100:
                self.discovery_history = self.discovery_history[-100:]
            
            if not result.success:
                return result
            
            # Filter out already registered devices
            new_devices = []
            existing_device_ids = set(self.iot_service.devices.keys())
            
            for device in result.discovered_devices:
                if device.device_id not in existing_device_ids:
                    new_devices.append(device)
            
            result.discovered_devices = new_devices
            
            # Auto-register new devices if enabled
            if auto_register and new_devices:
                registered_count = 0
                for device in new_devices:
                    if await self.register_device(device):
                        registered_count += 1
                
                logger.info(f"Auto-registered {registered_count}/{len(new_devices)} discovered devices")
            
            return result
            
        except Exception as e:
            logger.error(f"Device discovery and registration failed: {e}")
            return DeviceDiscoveryResult(
                discovered_devices=[],
                discovery_method="auto",
                success=False,
                error_message=str(e)
            )

    async def register_device(self, device: Device, validate: bool = True) -> bool:
        """Register a device with optional validation"""
        try:
            # Validate device if requested
            if validate:
                validation_result = await self.validate_device(device)
                if not validation_result["valid"]:
                    logger.warning(f"Device validation failed for {device.device_id}: {validation_result['errors']}")
                    return False
            
            # Register with IoT service
            success = await self.iot_service.register_device(device)
            
            if success:
                logger.info(f"Successfully registered device {device.device_id}")
                
                # Log registration event
                await self._log_device_event(device.device_id, "registered", {
                    "protocol": device.config.protocol.value,
                    "endpoint": device.config.endpoint,
                    "device_type": device.device_type
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register device {device.device_id}: {e}")
            return False

    async def unregister_device(self, device_id: str) -> bool:
        """Unregister a device"""
        try:
            success = await self.iot_service.unregister_device(device_id)
            
            if success:
                logger.info(f"Successfully unregistered device {device_id}")
                
                # Log unregistration event
                await self._log_device_event(device_id, "unregistered", {})
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to unregister device {device_id}: {e}")
            return False

    async def validate_device(self, device: Device) -> Dict[str, Any]:
        """Validate device configuration and connectivity"""
        errors = []
        warnings = []
        
        try:
            # Basic validation
            if not device.device_id or not device.device_id.strip():
                errors.append("Device ID cannot be empty")
            
            if not device.device_type or not device.device_type.strip():
                errors.append("Device type cannot be empty")
            
            if not device.config.endpoint or not device.config.endpoint.strip():
                errors.append("Endpoint cannot be empty")
            
            # Protocol-specific validation
            if device.config.protocol == ProtocolType.MQTT:
                if not device.config.endpoint.startswith(("mqtt://", "mqtts://")):
                    warnings.append("MQTT endpoint should start with mqtt:// or mqtts://")
            
            elif device.config.protocol == ProtocolType.HTTP_REST:
                if not device.config.endpoint.startswith(("http://", "https://")):
                    warnings.append("HTTP endpoint should start with http:// or https://")
            
            elif device.config.protocol == ProtocolType.MODBUS:
                if not device.config.endpoint.startswith(("modbus+tcp://", "modbus+rtu://")):
                    warnings.append("Modbus endpoint should start with modbus+tcp:// or modbus+rtu://")
            
            # Connectivity test
            connectivity_test = await self._test_device_connectivity(device)
            if not connectivity_test["success"]:
                warnings.append(f"Connectivity test failed: {connectivity_test['error']}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "connectivity": connectivity_test
            }
            
        except Exception as e:
            logger.error(f"Device validation failed for {device.device_id}: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": warnings,
                "connectivity": {"success": False, "error": str(e)}
            }

    async def get_registered_devices(self) -> List[Device]:
        """Get all registered devices"""
        return list(self.iot_service.devices.values())

    async def get_device(self, device_id: str) -> Optional[Device]:
        """Get a specific registered device"""
        return self.iot_service.devices.get(device_id)

    async def update_device(self, device_id: str, updates: Dict[str, Any]) -> bool:
        """Update device configuration"""
        try:
            if device_id not in self.iot_service.devices:
                logger.warning(f"Device {device_id} not found for update")
                return False
            
            device = self.iot_service.devices[device_id]
            
            # Update device fields
            for field, value in updates.items():
                if hasattr(device, field):
                    setattr(device, field, value)
                elif hasattr(device.config, field):
                    setattr(device.config, field, value)
            
            device.updated_at = datetime.utcnow()
            
            # Update in database
            await self._update_device_in_db(device)
            
            # Restart monitoring with new configuration
            await self.iot_service._stop_device_monitoring(device_id)
            await self.iot_service._start_device_monitoring(device_id)
            
            logger.info(f"Updated device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update device {device_id}: {e}")
            return False

    async def get_device_statistics(self) -> Dict[str, Any]:
        """Get device registry statistics"""
        try:
            devices = list(self.iot_service.devices.values())
            statuses = await self.iot_service.get_all_device_statuses()
            
            stats = {
                "total_devices": len(devices),
                "by_protocol": {},
                "by_status": {},
                "by_type": {},
                "discovery_history_count": len(self.discovery_history),
                "last_discovery": self.discovery_history[-1].timestamp if self.discovery_history else None
            }
            
            # Count by protocol
            for device in devices:
                protocol = device.config.protocol.value
                stats["by_protocol"][protocol] = stats["by_protocol"].get(protocol, 0) + 1
            
            # Count by status
            for status in statuses.values():
                status_str = status.value
                stats["by_status"][status_str] = stats["by_status"].get(status_str, 0) + 1
            
            # Count by type
            for device in devices:
                device_type = device.device_type
                stats["by_type"][device_type] = stats["by_type"].get(device_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get device statistics: {e}")
            return {}

    async def get_discovery_history(self, limit: int = 10) -> List[DeviceDiscoveryResult]:
        """Get recent discovery history"""
        return self.discovery_history[-limit:] if self.discovery_history else []

    async def load_devices_from_db(self):
        """Load registered devices from database on startup"""
        try:
            async with get_db_session() as session:
                result = await session.execute("SELECT * FROM devices")
                db_devices = result.fetchall()
                
                for db_device in db_devices:
                    device = DeviceDB(**dict(db_device)).to_pydantic()
                    await self.iot_service.register_device(device)
                
                logger.info(f"Loaded {len(db_devices)} devices from database")
                
        except Exception as e:
            logger.error(f"Failed to load devices from database: {e}")

    async def _test_device_connectivity(self, device: Device) -> Dict[str, Any]:
        """Test connectivity to a device"""
        try:
            # Create temporary handler for testing
            handler = self.iot_service._create_handler(device)
            if not handler:
                return {"success": False, "error": "Unsupported protocol"}
            
            # Test connection
            connected = await handler.connect()
            if connected:
                # Test ping
                ping_success = await handler.ping()
                await handler.disconnect()
                
                return {
                    "success": ping_success,
                    "error": None if ping_success else "Ping failed"
                }
            else:
                return {
                    "success": False,
                    "error": handler.last_error or "Connection failed"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _update_device_in_db(self, device: Device):
        """Update device in database"""
        try:
            async with get_db_session() as session:
                await session.execute(
                    "UPDATE devices SET name = ?, location = ?, config = ?, "
                    "status = ?, metadata = ?, updated_at = ? WHERE device_id = ?",
                    (device.name, device.location, json.dumps(device.config.model_dump()),
                     device.status.value, json.dumps(device.metadata) if device.metadata else None,
                     device.updated_at, device.device_id)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to update device {device.device_id} in database: {e}")

    async def _log_device_event(self, device_id: str, event_type: str, metadata: Dict[str, Any]):
        """Log device lifecycle events"""
        try:
            event = {
                "device_id": device_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
            
            # In a production system, this would go to a proper event log
            logger.info(f"Device event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Failed to log device event: {e}")