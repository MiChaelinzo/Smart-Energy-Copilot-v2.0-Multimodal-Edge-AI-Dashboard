"""
MQTT Protocol Handler

Handles MQTT communication with IoT devices.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import paho.mqtt.client as mqtt
from urllib.parse import urlparse

from .base import BaseProtocolHandler
from src.models.device import Device, DeviceStatus, ProtocolType, DeviceConfig
from src.models.sensor_reading import SensorReading, SensorReadings


class MQTTHandler(BaseProtocolHandler):
    """MQTT protocol handler for IoT devices"""

    def __init__(self, device: Device):
        super().__init__(device)
        self.client: Optional[mqtt.Client] = None
        self.data_topic = f"sensors/{device.device_id}/data"
        self.status_topic = f"sensors/{device.device_id}/status"
        self.latest_reading: Optional[SensorReading] = None
        self._connection_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            # Parse MQTT endpoint
            parsed = urlparse(self.device.config.endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or 1883

            # Create MQTT client
            self.client = mqtt.Client()
            
            # Set credentials if provided
            if self.device.config.credentials:
                username = self.device.config.credentials.get("username")
                password = self.device.config.credentials.get("password")
                if username and password:
                    self.client.username_pw_set(username, password)

            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Connect to broker
            self.client.connect_async(host, port, self.device.config.timeout)
            self.client.loop_start()

            # Wait for connection with timeout
            try:
                await asyncio.wait_for(self._connection_event.wait(), timeout=self.device.config.timeout)
                self._set_connected(True)
                self._clear_error()
                return True
            except asyncio.TimeoutError:
                self._set_error(f"Connection timeout to {host}:{port}")
                return False

        except Exception as e:
            self._set_error(f"MQTT connection failed: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.client = None
            self._set_connected(False)
            return True
        except Exception as e:
            self._set_error(f"MQTT disconnection failed: {str(e)}")
            return False

    async def read_data(self) -> Optional[SensorReading]:
        """Read latest sensor data from MQTT"""
        if not self.is_connected:
            return None
        
        # Return the latest reading received via MQTT
        return self.latest_reading

    async def ping(self) -> bool:
        """Ping device by publishing to status topic"""
        if not self.client or not self.is_connected:
            return False
        
        try:
            # Publish ping message
            ping_msg = {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
            result = self.client.publish(f"sensors/{self.device.device_id}/ping", json.dumps(ping_msg))
            
            # Check if publish was successful
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            self._set_error(f"MQTT ping failed: {str(e)}")
            return False

    async def discover_devices(self, broker_host: str = "localhost", broker_port: int = 1883, 
                             discovery_timeout: int = 30) -> List[Device]:
        """Discover MQTT devices by listening to discovery topic"""
        discovered = []
        
        try:
            # Create temporary client for discovery
            discovery_client = mqtt.Client()
            discovered_devices = []
            
            def on_discovery_message(client, userdata, message):
                try:
                    data = json.loads(message.payload.decode())
                    if data.get("type") == "device_announcement":
                        device_config = DeviceConfig(
                            protocol=ProtocolType.MQTT,
                            endpoint=f"mqtt://{broker_host}:{broker_port}",
                            polling_interval=data.get("polling_interval", 60),
                            timeout=30
                        )
                        
                        device = Device(
                            device_id=data["device_id"],
                            device_type=data.get("device_type", "unknown"),
                            name=data.get("name", data["device_id"]),
                            location=data.get("location", "unknown"),
                            config=device_config,
                            metadata=data.get("metadata", {})
                        )
                        discovered_devices.append(device)
                except Exception as e:
                    self.logger.warning(f"Failed to parse discovery message: {e}")
            
            discovery_client.on_message = on_discovery_message
            discovery_client.connect(broker_host, broker_port, 60)
            discovery_client.subscribe("sensors/+/discovery")
            discovery_client.loop_start()
            
            # Wait for discovery timeout
            await asyncio.sleep(discovery_timeout)
            
            discovery_client.loop_stop()
            discovery_client.disconnect()
            
            return discovered_devices
            
        except Exception as e:
            self.logger.error(f"MQTT device discovery failed: {e}")
            return []

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.logger.info(f"Connected to MQTT broker for device {self.device.device_id}")
            # Subscribe to data topic
            client.subscribe(self.data_topic)
            client.subscribe(self.status_topic)
            self._connection_event.set()
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
            self._set_error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.logger.info(f"Disconnected from MQTT broker for device {self.device.device_id}")
        self._set_connected(False)
        self._connection_event.clear()

    def _on_message(self, client, userdata, message):
        """MQTT message callback"""
        try:
            topic = message.topic
            payload = json.loads(message.payload.decode())
            
            if topic == self.data_topic:
                # Parse sensor reading
                readings_data = payload.get("readings", {})
                readings = SensorReadings(**readings_data)
                
                self.latest_reading = SensorReading(
                    sensor_id=self.device.device_id,
                    device_type=self.device.device_type,
                    timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.utcnow().isoformat())),
                    readings=readings,
                    quality_score=payload.get("quality_score", 1.0),
                    location=self.device.location
                )
                
                # Update device last seen
                self.device.last_seen = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")