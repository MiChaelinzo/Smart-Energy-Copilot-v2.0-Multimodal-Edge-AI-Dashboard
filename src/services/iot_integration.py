"""
IoT Integration Layer

Coordinates IoT protocol handlers, device discovery, data validation, and offline operation.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from collections import deque
import logging

from src.models.device import Device, DeviceStatus, ProtocolType, DeviceDiscoveryResult
from src.models.sensor_reading import SensorReading
from src.services.iot_protocols.base import BaseProtocolHandler
from src.services.iot_protocols.mqtt_handler import MQTTHandler
from src.services.iot_protocols.http_handler import HTTPHandler
from src.services.iot_protocols.modbus_handler import ModbusHandler
from src.database.connection import get_db_session
from src.models.device import DeviceDB
from src.models.sensor_reading import SensorReadingDB
from src.services.error_handling import (
    with_error_handling, IoTConnectionError, ValidationError,
    ErrorContext, ErrorSeverity, error_handler, retry_with_backoff
)

logger = logging.getLogger(__name__)


class IoTIntegrationService:
    """Main service for IoT device integration and management"""

    def __init__(self, max_buffer_size: int = 10000, offline_retention_hours: int = 24):
        self.devices: Dict[str, Device] = {}
        self.handlers: Dict[str, BaseProtocolHandler] = {}
        self.max_buffer_size = max_buffer_size
        self.offline_retention_hours = offline_retention_hours
        
        # Offline data buffering
        self.offline_buffer: deque = deque(maxlen=max_buffer_size)
        self.is_online = True
        
        # Device monitoring
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.reconnection_tasks: Dict[str, asyncio.Task] = {}
        
        # Data validation settings
        self.validation_rules = {
            "power_watts": {"min": 0, "max": 50000},
            "voltage": {"min": 0, "max": 500},
            "current_amps": {"min": 0, "max": 1000},
            "temperature_celsius": {"min": -50, "max": 100},
            "humidity_percent": {"min": 0, "max": 100}
        }

    @with_error_handling("iot_service", "register_device")
    @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
    async def register_device(self, device: Device) -> bool:
        """Register a new IoT device"""
        try:
            # Create appropriate protocol handler
            handler = self._create_handler(device)
            if not handler:
                raise IoTConnectionError(f"Unsupported protocol for device {device.device_id}: {device.config.protocol}")

            # Store device and handler
            self.devices[device.device_id] = device
            self.handlers[device.device_id] = handler

            # Persist to database
            await self._save_device_to_db(device)

            # Start monitoring
            await self._start_device_monitoring(device.device_id)

            logger.info(f"Registered device {device.device_id} with protocol {device.config.protocol}")
            return True

        except IoTConnectionError:
            # Re-raise IoT-specific errors
            raise
        except Exception as e:
            logger.error(f"Failed to register device {device.device_id}: {e}")
            raise IoTConnectionError(f"Device registration failed: {str(e)}")

    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device"""
        try:
            # Stop monitoring
            await self._stop_device_monitoring(device_id)

            # Disconnect handler
            if device_id in self.handlers:
                await self.handlers[device_id].disconnect()
                del self.handlers[device_id]

            # Remove device
            if device_id in self.devices:
                del self.devices[device_id]

            # Remove from database
            await self._remove_device_from_db(device_id)

            logger.info(f"Unregistered device {device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister device {device_id}: {e}")
            return False

    @with_error_handling("iot_service", "discover_devices")
    async def discover_devices(self, protocols: List[ProtocolType] = None, 
                             discovery_timeout: int = 30) -> DeviceDiscoveryResult:
        """Auto-discover IoT devices using specified protocols"""
        if protocols is None:
            protocols = [ProtocolType.MQTT, ProtocolType.HTTP_REST, ProtocolType.MODBUS]

        discovered_devices = []
        discovery_methods = []

        try:
            discovery_tasks = []
            
            for protocol in protocols:
                if protocol == ProtocolType.MQTT:
                    # Create temporary MQTT handler for discovery
                    temp_device = Device(
                        device_id="temp_mqtt",
                        device_type="temp",
                        name="temp",
                        location="temp",
                        config={"protocol": protocol, "endpoint": "mqtt://localhost:1883"}
                    )
                    handler = MQTTHandler(temp_device)
                    discovery_tasks.append(handler.discover_devices())
                    discovery_methods.append("MQTT")
                    
                elif protocol == ProtocolType.HTTP_REST:
                    temp_device = Device(
                        device_id="temp_http",
                        device_type="temp",
                        name="temp",
                        location="temp",
                        config={"protocol": protocol, "endpoint": "http://localhost"}
                    )
                    handler = HTTPHandler(temp_device)
                    discovery_tasks.append(handler.discover_devices())
                    discovery_methods.append("HTTP")
                    
                elif protocol == ProtocolType.MODBUS:
                    temp_device = Device(
                        device_id="temp_modbus",
                        device_type="temp",
                        name="temp",
                        location="temp",
                        config={"protocol": protocol, "endpoint": "modbus+tcp://localhost:502"}
                    )
                    handler = ModbusHandler(temp_device)
                    discovery_tasks.append(handler.discover_devices())
                    discovery_methods.append("Modbus")

            # Wait for all discovery tasks
            results = await asyncio.wait_for(
                asyncio.gather(*discovery_tasks, return_exceptions=True),
                timeout=discovery_timeout
            )

            # Collect discovered devices
            for i, result in enumerate(results):
                if isinstance(result, list):
                    discovered_devices.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"{discovery_methods[i]} discovery failed: {result}")

            return DeviceDiscoveryResult(
                discovered_devices=discovered_devices,
                discovery_method=", ".join(discovery_methods),
                success=True
            )

        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return DeviceDiscoveryResult(
                discovered_devices=[],
                discovery_method=", ".join(discovery_methods),
                success=False,
                error_message=str(e)
            )

    async def read_device_data(self, device_id: str) -> Optional[SensorReading]:
        """Read data from a specific device"""
        if device_id not in self.handlers:
            logger.warning(f"No handler found for device {device_id}")
            return None

        try:
            handler = self.handlers[device_id]
            reading = await handler.read_data()
            
            if reading:
                # Validate and interpolate data
                validated_reading = await self._validate_and_interpolate(reading)
                
                # Store reading
                await self._store_sensor_reading(validated_reading)
                
                return validated_reading
            else:
                logger.warning(f"No data received from device {device_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to read data from device {device_id}: {e}")
            return None

    async def read_all_devices(self) -> Dict[str, Optional[SensorReading]]:
        """Read data from all registered devices"""
        readings = {}
        
        tasks = []
        device_ids = list(self.devices.keys())
        
        for device_id in device_ids:
            tasks.append(self.read_device_data(device_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            device_id = device_ids[i]
            if isinstance(result, SensorReading):
                readings[device_id] = result
            elif isinstance(result, Exception):
                logger.error(f"Error reading from device {device_id}: {result}")
                readings[device_id] = None
            else:
                readings[device_id] = result
        
        return readings

    async def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        """Get current status of a device"""
        if device_id not in self.handlers:
            return None
        
        handler = self.handlers[device_id]
        return await handler.get_device_status()

    async def get_all_device_statuses(self) -> Dict[str, DeviceStatus]:
        """Get status of all registered devices"""
        statuses = {}
        
        for device_id, handler in self.handlers.items():
            try:
                status = await handler.get_device_status()
                statuses[device_id] = status
            except Exception as e:
                logger.error(f"Error getting status for device {device_id}: {e}")
                statuses[device_id] = DeviceStatus.ERROR
        
        return statuses

    async def start_monitoring(self):
        """Start monitoring all registered devices"""
        for device_id in self.devices.keys():
            await self._start_device_monitoring(device_id)

    async def stop_monitoring(self):
        """Stop monitoring all devices"""
        for device_id in list(self.monitoring_tasks.keys()):
            await self._stop_device_monitoring(device_id)

    async def flush_offline_buffer(self) -> int:
        """Flush offline buffer to database when connection is restored"""
        if not self.offline_buffer:
            return 0

        flushed_count = 0
        try:
            async with get_db_session() as session:
                while self.offline_buffer:
                    reading = self.offline_buffer.popleft()
                    db_reading = SensorReadingDB.from_pydantic(reading)
                    session.add(db_reading)
                    flushed_count += 1
                
                await session.commit()
                logger.info(f"Flushed {flushed_count} readings from offline buffer")
                
        except Exception as e:
            logger.error(f"Failed to flush offline buffer: {e}")
            
        return flushed_count

    def _create_handler(self, device: Device) -> Optional[BaseProtocolHandler]:
        """Create appropriate protocol handler for device"""
        protocol = device.config.protocol
        
        if protocol == ProtocolType.MQTT:
            return MQTTHandler(device)
        elif protocol == ProtocolType.HTTP_REST:
            return HTTPHandler(device)
        elif protocol == ProtocolType.MODBUS:
            return ModbusHandler(device)
        else:
            return None

    async def _validate_and_interpolate(self, reading: SensorReading) -> SensorReading:
        """Validate sensor data and interpolate missing/invalid values"""
        validated_readings = {}
        
        for field, value in reading.readings.model_dump(exclude_none=True).items():
            if field in self.validation_rules:
                rules = self.validation_rules[field]
                
                # Check bounds
                if value < rules["min"] or value > rules["max"]:
                    logger.warning(f"Invalid {field} value {value} for device {reading.sensor_id}")
                    # Interpolate with last known good value or default
                    interpolated_value = await self._interpolate_value(reading.sensor_id, field)
                    validated_readings[field] = interpolated_value
                    reading.quality_score *= 0.8  # Reduce quality score
                else:
                    validated_readings[field] = value
            else:
                validated_readings[field] = value

        # Update readings with validated values
        from src.models.sensor_reading import SensorReadings
        reading.readings = SensorReadings(**validated_readings)
        
        return reading

    async def _interpolate_value(self, sensor_id: str, field: str) -> float:
        """Interpolate missing or invalid sensor value"""
        try:
            # Get last known good value from database
            async with get_db_session() as session:
                last_reading = await session.execute(
                    f"SELECT readings FROM sensor_readings WHERE sensor_id = '{sensor_id}' "
                    f"AND readings ->> '{field}' IS NOT NULL "
                    f"ORDER BY timestamp DESC LIMIT 1"
                )
                result = last_reading.fetchone()
                
                if result:
                    last_readings = json.loads(result[0])
                    return float(last_readings.get(field, 0))
                else:
                    # Return default value based on field type
                    defaults = {
                        "power_watts": 0.0,
                        "voltage": 240.0,
                        "current_amps": 0.0,
                        "temperature_celsius": 20.0,
                        "humidity_percent": 50.0
                    }
                    return defaults.get(field, 0.0)
                    
        except Exception as e:
            logger.error(f"Failed to interpolate value for {sensor_id}.{field}: {e}")
            return 0.0

    async def _start_device_monitoring(self, device_id: str):
        """Start monitoring task for a device"""
        if device_id in self.monitoring_tasks:
            return  # Already monitoring

        async def monitor_device():
            device = self.devices[device_id]
            handler = self.handlers[device_id]
            
            while device_id in self.devices:
                try:
                    # Check if device is connected
                    if not handler.is_connected:
                        # Try to reconnect
                        await self._attempt_reconnection(device_id)
                    
                    # Read data if connected
                    if handler.is_connected:
                        await self.read_device_data(device_id)
                    
                    # Wait for next polling interval
                    await asyncio.sleep(device.config.polling_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error monitoring device {device_id}: {e}")
                    await asyncio.sleep(device.config.polling_interval)

        task = asyncio.create_task(monitor_device())
        self.monitoring_tasks[device_id] = task

    async def _stop_device_monitoring(self, device_id: str):
        """Stop monitoring task for a device"""
        if device_id in self.monitoring_tasks:
            task = self.monitoring_tasks[device_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.monitoring_tasks[device_id]

        if device_id in self.reconnection_tasks:
            task = self.reconnection_tasks[device_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.reconnection_tasks[device_id]

    async def _attempt_reconnection(self, device_id: str):
        """Attempt to reconnect to a device"""
        if device_id in self.reconnection_tasks:
            return  # Already attempting reconnection

        async def reconnect():
            device = self.devices[device_id]
            handler = self.handlers[device_id]
            
            for attempt in range(device.config.retry_attempts):
                try:
                    logger.info(f"Reconnection attempt {attempt + 1} for device {device_id}")
                    
                    if await handler.connect():
                        logger.info(f"Successfully reconnected to device {device_id}")
                        return
                    
                    # Wait before next attempt
                    await asyncio.sleep(min(2 ** attempt, 60))  # Exponential backoff
                    
                except Exception as e:
                    logger.error(f"Reconnection attempt {attempt + 1} failed for device {device_id}: {e}")
            
            logger.error(f"Failed to reconnect to device {device_id} after {device.config.retry_attempts} attempts")

        task = asyncio.create_task(reconnect())
        self.reconnection_tasks[device_id] = task
        
        try:
            await task
        finally:
            if device_id in self.reconnection_tasks:
                del self.reconnection_tasks[device_id]

    async def _save_device_to_db(self, device: Device):
        """Save device to database"""
        try:
            async with get_db_session() as session:
                db_device = DeviceDB.from_pydantic(device)
                session.add(db_device)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save device {device.device_id} to database: {e}")

    async def _remove_device_from_db(self, device_id: str):
        """Remove device from database"""
        try:
            async with get_db_session() as session:
                await session.execute(f"DELETE FROM devices WHERE device_id = '{device_id}'")
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to remove device {device_id} from database: {e}")

    async def _store_sensor_reading(self, reading: SensorReading):
        """Store sensor reading to database or offline buffer"""
        try:
            if self.is_online:
                async with get_db_session() as session:
                    db_reading = SensorReadingDB.from_pydantic(reading)
                    session.add(db_reading)
                    await session.commit()
            else:
                # Store in offline buffer
                self.offline_buffer.append(reading)
                logger.debug(f"Stored reading in offline buffer for device {reading.sensor_id}")
                
        except Exception as e:
            logger.error(f"Failed to store sensor reading: {e}")
            # Fallback to offline buffer
            self.offline_buffer.append(reading)

    async def set_online_status(self, is_online: bool):
        """Set online/offline status for the service"""
        self.is_online = is_online
        
        if is_online:
            # Flush offline buffer when coming back online
            await self.flush_offline_buffer()