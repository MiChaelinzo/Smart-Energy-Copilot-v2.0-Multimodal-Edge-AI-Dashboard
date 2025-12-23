#!/usr/bin/env python3

import sys

# Clear any cached modules
modules_to_clear = [k for k in sys.modules.keys() if 'modbus' in k.lower()]
for module in modules_to_clear:
    del sys.modules[module]

from src.services.iot_protocols.base import BaseProtocolHandler
from src.models.device import Device, DeviceConfig, ProtocolType

# Create a simple test class
class TestModbusHandler(BaseProtocolHandler):
    def __init__(self, device):
        print("TestModbusHandler.__init__ called")
        super().__init__(device)
        print("super().__init__ completed")
        self.client = None
        print("client set")
        self.slave_id = device.config.credentials.get("slave_id", 1) if device.config.credentials else 1
        print(f"slave_id set to: {self.slave_id}")
        self.register_map = device.metadata.get("register_map", {}) if device.metadata else {}
        print(f"register_map set to: {self.register_map}")
        print("TestModbusHandler.__init__ completed")
    
    async def connect(self):
        return False
    
    async def disconnect(self):
        return True
    
    async def read_data(self):
        return None
    
    async def ping(self):
        return False
    
    async def discover_devices(self, **kwargs):
        return []

device = Device(
    device_id='test',
    device_type='test', 
    name='test',
    location='test',
    config=DeviceConfig(
        protocol=ProtocolType.MODBUS,
        endpoint='modbus+tcp://localhost:502',
        credentials=None
    )
)

print("Creating TestModbusHandler...")
handler = TestModbusHandler(device)
print("Handler created")
print("Has slave_id:", hasattr(handler, 'slave_id'))
print("Has register_map:", hasattr(handler, 'register_map'))
if hasattr(handler, 'slave_id'):
    print("slave_id value:", handler.slave_id)
if hasattr(handler, 'register_map'):
    print("register_map value:", handler.register_map)