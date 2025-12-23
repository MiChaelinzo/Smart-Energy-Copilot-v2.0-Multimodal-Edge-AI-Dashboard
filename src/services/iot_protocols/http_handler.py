"""
HTTP REST Protocol Handler

Handles HTTP REST API communication with IoT devices.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
from urllib.parse import urljoin

from .base import BaseProtocolHandler
from src.models.device import Device, DeviceStatus, ProtocolType, DeviceConfig
from src.models.sensor_reading import SensorReading, SensorReadings


class HTTPHandler(BaseProtocolHandler):
    """HTTP REST protocol handler for IoT devices"""

    def __init__(self, device: Device):
        super().__init__(device)
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = self.device.config.endpoint.rstrip('/')

    async def connect(self) -> bool:
        """Initialize HTTP session"""
        try:
            # Create session with timeout
            timeout = aiohttp.ClientTimeout(total=self.device.config.timeout)
            
            # Set up authentication headers if credentials provided
            headers = {}
            if self.device.config.credentials:
                auth_type = self.device.config.credentials.get("auth_type", "basic")
                if auth_type == "basic":
                    username = self.device.config.credentials.get("username")
                    password = self.device.config.credentials.get("password")
                    if username and password:
                        auth = aiohttp.BasicAuth(username, password)
                        self.session = aiohttp.ClientSession(timeout=timeout, auth=auth)
                    else:
                        self.session = aiohttp.ClientSession(timeout=timeout)
                elif auth_type == "bearer":
                    token = self.device.config.credentials.get("token")
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
                    else:
                        self.session = aiohttp.ClientSession(timeout=timeout)
                elif auth_type == "api_key":
                    api_key = self.device.config.credentials.get("api_key")
                    key_header = self.device.config.credentials.get("key_header", "X-API-Key")
                    if api_key:
                        headers[key_header] = api_key
                        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
                    else:
                        self.session = aiohttp.ClientSession(timeout=timeout)
                else:
                    self.session = aiohttp.ClientSession(timeout=timeout)
            else:
                self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection with ping
            if await self.ping():
                self._set_connected(True)
                self._clear_error()
                return True
            else:
                self._set_error("HTTP connection test failed")
                return False

        except Exception as e:
            self._set_error(f"HTTP connection failed: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """Close HTTP session"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self._set_connected(False)
            return True
        except Exception as e:
            self._set_error(f"HTTP disconnection failed: {str(e)}")
            return False

    async def read_data(self) -> Optional[SensorReading]:
        """Read sensor data via HTTP GET request"""
        if not self.session or not self.is_connected:
            return None

        try:
            # Make GET request to data endpoint
            data_url = urljoin(self.base_url, "/data")
            async with self.session.get(data_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse response into SensorReading
                    readings_data = data.get("readings", {})
                    readings = SensorReadings(**readings_data)
                    
                    sensor_reading = SensorReading(
                        sensor_id=self.device.device_id,
                        device_type=self.device.device_type,
                        timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
                        readings=readings,
                        quality_score=data.get("quality_score", 1.0),
                        location=self.device.location
                    )
                    
                    # Update device last seen
                    self.device.last_seen = datetime.utcnow()
                    return sensor_reading
                else:
                    self._set_error(f"HTTP request failed with status {response.status}")
                    return None

        except Exception as e:
            self._set_error(f"HTTP read data failed: {str(e)}")
            return None

    async def ping(self) -> bool:
        """Ping device via HTTP health check"""
        if not self.session:
            return False

        try:
            # Try health check endpoint first
            health_url = urljoin(self.base_url, "/health")
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    return True
                    
            # Fallback to status endpoint
            status_url = urljoin(self.base_url, "/status")
            async with self.session.get(status_url) as response:
                if response.status == 200:
                    return True
                    
            # Fallback to root endpoint
            async with self.session.get(self.base_url) as response:
                return response.status < 500  # Accept any non-server error
                
        except Exception as e:
            self.logger.debug(f"HTTP ping failed: {e}")
            return False

    async def discover_devices(self, network_range: str = "192.168.1.0/24", 
                             common_ports: List[int] = None, 
                             discovery_timeout: int = 30) -> List[Device]:
        """Discover HTTP devices by scanning network"""
        if common_ports is None:
            common_ports = [80, 8080, 8000, 3000, 5000]
            
        discovered = []
        
        try:
            # Simple network scanning (in production, use proper network discovery)
            import ipaddress
            network = ipaddress.IPv4Network(network_range, strict=False)
            
            # Create session for discovery
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                
                tasks = []
                for ip in network.hosts():
                    for port in common_ports:
                        tasks.append(self._check_http_device(session, str(ip), port))
                
                # Wait for all checks with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=discovery_timeout
                    )
                    
                    # Collect successful discoveries
                    for result in results:
                        if isinstance(result, Device):
                            discovered.append(result)
                            
                except asyncio.TimeoutError:
                    self.logger.warning("HTTP device discovery timed out")
                    
        except Exception as e:
            self.logger.error(f"HTTP device discovery failed: {e}")
            
        return discovered

    async def _check_http_device(self, session: aiohttp.ClientSession, 
                                ip: str, port: int) -> Optional[Device]:
        """Check if HTTP device exists at given IP:port"""
        try:
            url = f"http://{ip}:{port}"
            
            # Try to get device info
            async with session.get(f"{url}/device/info") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    device_config = DeviceConfig(
                        protocol=ProtocolType.HTTP_REST,
                        endpoint=url,
                        polling_interval=60,
                        timeout=30
                    )
                    
                    device = Device(
                        device_id=data.get("device_id", f"http_{ip}_{port}"),
                        device_type=data.get("device_type", "http_device"),
                        name=data.get("name", f"HTTP Device {ip}:{port}"),
                        location=data.get("location", "unknown"),
                        config=device_config,
                        metadata=data.get("metadata", {"ip": ip, "port": port})
                    )
                    return device
                    
        except Exception:
            # Silently ignore failed connections during discovery
            pass
            
        return None