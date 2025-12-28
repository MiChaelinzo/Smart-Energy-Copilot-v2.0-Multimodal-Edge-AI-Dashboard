/**
 * Energy Service for mobile app
 * Handles energy consumption data and forecasting
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

export interface EnergyData {
  currentConsumption: number;
  dailyConsumption: number;
  monthlyConsumption: number;
  estimatedCost: number;
  savingsThisMonth: number;
}

export interface ConsumptionHistory {
  timestamp: string;
  consumption: number;
  cost: number;
}

export interface ForecastData {
  timestamp: string;
  predictedConsumption: number;
  predictedCost: number;
  confidenceScore: number;
}

class EnergyService {
  private baseUrl: string;
  private isInitialized: boolean = false;

  constructor() {
    this.baseUrl = 'http://localhost:8000/api'; // Default backend URL
  }

  async initialize(): Promise<void> {
    try {
      // Load configuration from storage
      const savedUrl = await AsyncStorage.getItem('backend_url');
      if (savedUrl) {
        this.baseUrl = savedUrl;
      }

      this.isInitialized = true;
      console.log('Energy Service initialized');
    } catch (error) {
      console.error('Energy Service initialization error:', error);
    }
  }

  async getCurrentStatus(): Promise<EnergyData> {
    try {
      const response = await axios.get(`${this.baseUrl}/energy/status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching energy status:', error);
      // Return mock data for offline mode
      return {
        currentConsumption: 2.5,
        dailyConsumption: 45.2,
        monthlyConsumption: 1250,
        estimatedCost: 125.50,
        savingsThisMonth: 23.75,
      };
    }
  }

  async getConsumptionHistory(days: number = 7): Promise<ConsumptionHistory[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/energy/history`, {
        params: { days }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching consumption history:', error);
      // Return mock data
      return Array.from({ length: days }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
        consumption: 40 + Math.random() * 20,
        cost: (40 + Math.random() * 20) * 0.12,
      }));
    }
  }

  async getConsumptionByCategory(): Promise<Array<{ name: string; consumption: number; color: string }>> {
    try {
      const response = await axios.get(`${this.baseUrl}/energy/by-category`);
      return response.data;
    } catch (error) {
      console.error('Error fetching consumption by category:', error);
      // Return mock data
      return [
        { name: 'HVAC', consumption: 45, color: '#FF6B6B' },
        { name: 'Lighting', consumption: 20, color: '#4ECDC4' },
        { name: 'Appliances', consumption: 25, color: '#45B7D1' },
        { name: 'Electronics', consumption: 10, color: '#96CEB4' },
      ];
    }
  }

  async getForecast(hours: number = 24): Promise<ForecastData[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/energy/forecast`, {
        params: { hours }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching forecast:', error);
      // Return mock forecast data
      return Array.from({ length: hours }, (_, i) => ({
        timestamp: new Date(Date.now() + i * 60 * 60 * 1000).toISOString(),
        predictedConsumption: 45 + Math.sin(i / 4) * 10,
        predictedCost: (45 + Math.sin(i / 4) * 10) * 0.12,
        confidenceScore: 0.85 + Math.random() * 0.1,
      }));
    }
  }

  async uploadBill(imageUri: string): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'utility_bill.jpg',
      } as any);

      const response = await axios.post(`${this.baseUrl}/ocr/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return { success: true, data: response.data };
    } catch (error) {
      console.error('Error uploading bill:', error);
      return { success: false, error: 'Failed to upload bill' };
    }
  }

  async getAnomalies(): Promise<Array<{ timestamp: string; severity: string; message: string }>> {
    try {
      const response = await axios.get(`${this.baseUrl}/energy/anomalies`);
      return response.data;
    } catch (error) {
      console.error('Error fetching anomalies:', error);
      return [];
    }
  }

  async setBackendUrl(url: string): Promise<void> {
    this.baseUrl = url;
    await AsyncStorage.setItem('backend_url', url);
  }
}

export default new EnergyService();