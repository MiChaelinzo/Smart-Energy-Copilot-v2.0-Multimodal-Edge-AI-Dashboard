#!/usr/bin/env python3

from src.models.device import Device, DeviceConfig, ProtocolType
from src.services.iot_protocols.modbus_handler import ModbusHandler

def test_modbus_handler():
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
    
    print("Creating ModbusHandler...")
    handler = ModbusHandler(device)
    
    print("Handler created")
    print("Handler type:", type(handler))
    print("Handler MRO:", type(handler).__mro__)
    
    # Check attributes
    attrs = ['device', 'client', 'slave_id', 'register_map', 'is_connected', 'last_error']
    for attr in attrs:
        has_attr = hasattr(handler, attr)
        print(f"Has {attr}: {has_attr}")
        if has_attr:
            try:
                value = getattr(handler, attr)
                print(f"  Value: {value}")
            except Exception as e:
                print(f"  Error getting value: {e}")

if __name__ == "__main__":
    test_modbus_handler()