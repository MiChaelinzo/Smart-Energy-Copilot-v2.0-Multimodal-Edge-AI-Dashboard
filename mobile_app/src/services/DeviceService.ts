/**
 * Device Service for mobile app
 * Handles smart home device management and control
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

export interface SmartDevice {
  deviceId: string;
  name: string;
  type: string;
  protocol: string;
  isOnline: boolean;
  energyConsumption: number;
  room: string;
  capabilities: string[];
  currentState: Record<string, any>;
}

export interface DeviceStatus {
  activeDevices: number;
  totalDevices: number;
  totalConsumption: number;
  devicesByType: Record<string, number>;
  devicesByRoom: Record<string, number>;
}

class DeviceService {
  private baseUrl: string;
  private isInitialized: boolean = false;

  constructor() {
    this.baseUrl = 'http://localhost:8000/api';
  }

  async initialize(): Promise<void> {
    try {
      const savedUrl = await AsyncStorage.getItem('backend_url');
      if (savedUrl) {
        this.baseUrl = savedUrl;
      }

      this.isInitialized = true;
      console.log('Device Service initialized');
    } catch (error) {
      console.error('Device Service initialization error:', error);
    }
  }

  async getDeviceStatus(): Promise<DeviceStatus> {
    try {
      const response = await axios.get(`${this.baseUrl}/devices/status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching device status:', error);
      // Return mock data
      return {
        activeDevices: 12,
        totalDevices: 15,
        totalConsumption: 2.8,
        devicesByType: {
          lights: 6,
          thermostats: 2,
          plugs: 4,
          sensors: 3,
        },
        devicesByRoom: {
          'Living Room': 4,
          'Kitchen': 3,
          'Bedroom': 5,
          'Bathroom': 3,
        },
      };
    }
  }

  async getDevices(): Promise<SmartDevice[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/devices`);
      return response.data;
    } catch (error) {
      console.error('Error fetching devices:', error);
      // Return mock devices
      return [
        {
          deviceId: 'light_001',
          name: 'Living Room Light',
          type: 'light',
          protocol: 'zigbee',
          isOnline: true,
          energyConsumption: 9.5,
          room: 'Living Room',
          capabilities: ['on_off', 'brightness', 'color'],
          currentState: { power: 'on', brightness: 80 },
        },
        {
          deviceId: 'thermostat_001',
          name: 'Main Thermostat',
          type: 'thermostat',
          protocol: 'zwave',
          isOnline: true,
          energyConsumption: 3.2,
          room: 'Hallway',
          capabilities: ['temperature_control', 'scheduling'],
          currentState: { temperature: 22, targetTemp: 21 },
        },
        {
          deviceId: 'plug_001',
          name: 'Office Smart Plug',
          type: 'smart_plug',
          protocol: 'wifi',
          isOnline: false,
          energyConsumption: 0,
          room: 'Office',
          capabilities: ['on_off', 'energy_monitoring'],
          currentState: { power: 'off' },
        },
      ];
    }
  }

  async controlDevice(deviceId: string, command: string, parameters?: Record<string, any>): Promise<boolean> {
    try {
      const response = await axios.post(`${this.baseUrl}/devices/${deviceId}/control`, {
        command,
        parameters: parameters || {},
      });
      return response.data.success;
    } catch (error) {
      console.error('Error controlling device:', error);
      return false;
    }
  }

  async discoverDevices(): Promise<SmartDevice[]> {
    try {
      const response = await axios.post(`${this.baseUrl}/devices/discover`);
      return response.data;
    } catch (error) {
      console.error('Error discovering devices:', error);
      return [];
    }
  }

  async getScenes(): Promise<Array<{ id: string; name: string; description: string; isActive: boolean }>> {
    try {
      const response = await axios.get(`${this.baseUrl}/scenes`);
      return response.data;
    } catch (error) {
      console.error('Error fetching scenes:', error);
      // Return mock scenes
      return [
        {
          id: 'energy_saving',
          name: 'Energy Saving',
          description: 'Optimize all devices for energy efficiency',
          isActive: false,
        },
        {
          id: 'comfort_mode',
          name: 'Comfort Mode',
          description: 'Balance comfort and efficiency',
          isActive: true,
        },
        {
          id: 'away_mode',
          name: 'Away Mode',
          description: 'Minimal energy usage when away',
          isActive: false,
        },
      ];
    }
  }

  async activateScene(sceneId: string): Promise<boolean> {
    try {
      const response = await axios.post(`${this.baseUrl}/scenes/${sceneId}/activate`);
      return response.data.success;
    } catch (error) {
      console.error('Error activating scene:', error);
      return false;
    }
  }

  async getAutomationRules(): Promise<Array<{ id: string; name: string; isEnabled: boolean; description: string }>> {
    try {
      const response = await axios.get(`${this.baseUrl}/automation/rules`);
      return response.data;
    } catch (error) {
      console.error('Error fetching automation rules:', error);
      // Return mock rules
      return [
        {
          id: 'auto_lights',
          name: 'Auto Lights',
          isEnabled: true,
          description: 'Turn lights on/off based on occupancy',
        },
        {
          id: 'energy_optimizer',
          name: 'Energy Optimizer',
          isEnabled: true,
          description: 'Automatically optimize device settings for energy savings',
        },
        {
          id: 'night_mode',
          name: 'Night Mode',
          isEnabled: false,
          description: 'Reduce energy consumption during night hours',
        },
      ];
    }
  }

  async toggleAutomationRule(ruleId: string, enabled: boolean): Promise<boolean> {
    try {
      const response = await axios.patch(`${this.baseUrl}/automation/rules/${ruleId}`, {
        enabled,
      });
      return response.data.success;
    } catch (error) {
      console.error('Error toggling automation rule:', error);
      return false;
    }
  }
}

export default new DeviceService();