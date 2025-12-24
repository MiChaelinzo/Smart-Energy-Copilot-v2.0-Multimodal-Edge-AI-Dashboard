"""
Base Protocol Handler

Abstract base class for IoT protocol handlers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.models.device import Device, DeviceStatus
from src.models.sensor_reading import SensorReading

logger = logging.getLogger(__name__)


class BaseProtocolHandler(ABC):
    """Abstract base class for IoT protocol handlers"""

    def __init__(self, device: Device):
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._connected = False
        self._last_error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if handler is connected to device"""
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message"""
        return self._last_error

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the IoT device
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the IoT device
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def read_data(self) -> Optional[SensorReading]:
        """
        Read sensor data from the device
        
        Returns:
            Optional[SensorReading]: Sensor reading if successful, None otherwise
        """
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """
        Ping the device to check connectivity
        
        Returns:
            bool: True if device responds, False otherwise
        """
        pass

    @abstractmethod
    async def discover_devices(self, **kwargs) -> List[Device]:
        """
        Discover devices using this protocol
        
        Returns:
            List[Device]: List of discovered devices
        """
        pass

    async def send_command(self, command: Dict[str, Any]) -> bool:
        """
        Send command to the device (optional, not all protocols support this)
        
        Args:
            command: Command dictionary to send to device
            
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        self.logger.warning(f"Command sending not implemented for {self.__class__.__name__}")
        return False

    async def get_device_status(self) -> DeviceStatus:
        """
        Get current device status
        
        Returns:
            DeviceStatus: Current status of the device
        """
        try:
            if await self.ping():
                return DeviceStatus.ONLINE
            else:
                return DeviceStatus.OFFLINE
        except Exception as e:
            self.logger.error(f"Error checking device status: {e}")
            self._last_error = str(e)
            return DeviceStatus.ERROR

    def _set_connected(self, connected: bool):
        """Set connection status"""
        self._connected = connected
        if connected:
            self.device.last_seen = datetime.utcnow()
            self.device.status = DeviceStatus.ONLINE
        else:
            self.device.status = DeviceStatus.OFFLINE

    def _set_error(self, error_message: str):
        """Set error message and update device status"""
        self._last_error = error_message
        self.device.status = DeviceStatus.ERROR
        self._connected = False
        self.logger.error(f"Device {self.device.device_id}: {error_message}")

    def _clear_error(self):
        """Clear error state"""
        self._last_error = None