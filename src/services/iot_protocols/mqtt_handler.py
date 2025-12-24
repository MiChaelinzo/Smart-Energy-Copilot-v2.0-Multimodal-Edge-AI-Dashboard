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
        self.command_topic = f"sensors/{device.device_id}/command"
        self.latest_reading: Optional[SensorReading] = None
        self._connection_event = asyncio.Event()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = device.config.retry_attempts
        self._reconnect_delay = 1  # Start with 1 second delay

    async def connect(self) -> bool:
        """Connect to MQTT broker with retry logic"""
        for attempt in range(self._max_reconnect_attempts):
            try:
                # Parse MQTT endpoint
                parsed = urlparse(self.device.config.endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 1883

                # Create MQTT client with unique client ID
                client_id = f"{self.device.device_id}_{datetime.now().timestamp()}"
                self.client = mqtt.Client(client_id=client_id)
                
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
                self.client.on_log = self._on_log

                # Set connection options
                self.client.reconnect_delay_set(min_delay=1, max_delay=120)
                
                # Connect to broker
                self.client.connect_async(host, port, self.device.config.timeout)
                self.client.loop_start()

                # Wait for connection with timeout
                try:
                    await asyncio.wait_for(self._connection_event.wait(), timeout=self.device.config.timeout)
                    self._set_connected(True)
                    self._clear_error()
                    self._reconnect_attempts = 0  # Reset on successful connection
                    self.logger.info(f"MQTT connected to {host}:{port} on attempt {attempt + 1}")
                    return True
                except asyncio.TimeoutError:
                    error_msg = f"Connection timeout to {host}:{port} (attempt {attempt + 1})"
                    self.logger.warning(error_msg)
                    if attempt < self._max_reconnect_attempts - 1:
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(self._reconnect_delay * 2, 60)  # Exponential backoff
                    else:
                        self._set_error(error_msg)
                        return False

            except Exception as e:
                error_msg = f"MQTT connection failed: {str(e)} (attempt {attempt + 1})"
                self.logger.warning(error_msg)
                if attempt < self._max_reconnect_attempts - 1:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 60)
                else:
                    self._set_error(error_msg)
                    return False
        
        return False

    async def disconnect(self) -> bool:
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                # Wait a bit for clean disconnection
                await asyncio.sleep(0.1)
                self.client = None
            self._set_connected(False)
            self._connection_event.clear()
            self.logger.info(f"MQTT disconnected from device {self.device.device_id}")
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
        """Ping device by publishing to status topic and checking connection"""
        if not self.client or not self.is_connected:
            return False
        
        try:
            # First check if client is still connected
            if not self.client.is_connected():
                self._set_connected(False)
                return False
            
            # Publish ping message
            ping_msg = {
                "type": "ping", 
                "timestamp": datetime.utcnow().isoformat(),
                "device_id": self.device.device_id
            }
            result = self.client.publish(f"sensors/{self.device.device_id}/ping", json.dumps(ping_msg))
            
            # Check if publish was successful
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                # Wait for message to be sent
                result.wait_for_publish(timeout=5.0)
                return True
            else:
                self.logger.warning(f"MQTT ping publish failed with code {result.rc}")
                return False
                
        except Exception as e:
            self._set_error(f"MQTT ping failed: {str(e)}")
            return False

    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to MQTT device"""
        if not self.client or not self.is_connected:
            return False
        
        try:
            command_msg = {
                "command": command,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "energy_copilot"
            }
            result = self.client.publish(self.command_topic, json.dumps(command_msg), qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                result.wait_for_publish(timeout=5.0)
                self.logger.info(f"Command sent to device {self.device.device_id}: {command}")
                return True
            else:
                self.logger.error(f"Failed to send command to device {self.device.device_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending command to device {self.device.device_id}: {e}")
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
            # Subscribe to data and status topics
            client.subscribe(self.data_topic, qos=1)
            client.subscribe(self.status_topic, qos=1)
            # Also subscribe to command topic for device control
            client.subscribe(self.command_topic, qos=1)
            self._connection_event.set()
        else:
            error_codes = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_codes.get(rc, f"Connection failed with code {rc}")
            self.logger.error(f"MQTT connection failed: {error_msg}")
            self._set_error(f"MQTT connection failed: {error_msg}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        if rc != 0:
            self.logger.warning(f"Unexpected MQTT disconnection for device {self.device.device_id}, code: {rc}")
        else:
            self.logger.info(f"MQTT disconnected from device {self.device.device_id}")
        self._set_connected(False)
        self._connection_event.clear()

    def _on_log(self, client, userdata, level, buf):
        """MQTT logging callback"""
        if level == mqtt.MQTT_LOG_ERR:
            self.logger.error(f"MQTT: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            self.logger.warning(f"MQTT: {buf}")
        else:
            self.logger.debug(f"MQTT: {buf}")

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
                self.logger.debug(f"Received data from MQTT device {self.device.device_id}")
                
            elif topic == self.status_topic:
                # Handle status updates
                status = payload.get("status", "unknown")
                if status == "online":
                    self.device.status = DeviceStatus.ONLINE
                elif status == "offline":
                    self.device.status = DeviceStatus.OFFLINE
                else:
                    self.device.status = DeviceStatus.UNKNOWN
                self.logger.debug(f"Device {self.device.device_id} status: {status}")
                
            elif topic == self.command_topic:
                # Handle command responses
                self.logger.debug(f"Command response from device {self.device.device_id}: {payload}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in MQTT message from {topic}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing MQTT message from {topic}: {e}")