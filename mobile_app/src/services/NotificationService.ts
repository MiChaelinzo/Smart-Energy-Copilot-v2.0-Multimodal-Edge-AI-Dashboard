/**
 * Notification Service for mobile app
 * Handles push notifications and alerts
 */

import PushNotification from 'react-native-push-notification';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface Alert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  isRead: boolean;
}

class NotificationService {
  private isInitialized: boolean = false;
  private alerts: Alert[] = [];

  async initialize(): Promise<void> {
    try {
      // Configure push notifications
      PushNotification.configure({
        onRegister: (token) => {
          console.log('Push notification token:', token);
          this.savePushToken(token.token);
        },
        onNotification: (notification) => {
          console.log('Notification received:', notification);
          this.handleNotification(notification);
        },
        permissions: {
          alert: true,
          badge: true,
          sound: true,
        },
        popInitialNotification: true,
        requestPermissions: true,
      });

      // Load saved alerts
      await this.loadAlerts();

      this.isInitialized = true;
      console.log('Notification Service initialized');
    } catch (error) {
      console.error('Notification Service initialization error:', error);
    }
  }

  private async savePushToken(token: string): Promise<void> {
    try {
      await AsyncStorage.setItem('push_token', token);
      // Send token to backend
      // await this.sendTokenToBackend(token);
    } catch (error) {
      console.error('Error saving push token:', error);
    }
  }

  private async loadAlerts(): Promise<void> {
    try {
      const savedAlerts = await AsyncStorage.getItem('alerts');
      if (savedAlerts) {
        this.alerts = JSON.parse(savedAlerts).map((alert: any) => ({
          ...alert,
          timestamp: new Date(alert.timestamp),
        }));
      }
    } catch (error) {
      console.error('Error loading alerts:', error);
    }
  }

  private async saveAlerts(): Promise<void> {
    try {
      await AsyncStorage.setItem('alerts', JSON.stringify(this.alerts));
    } catch (error) {
      console.error('Error saving alerts:', error);
    }
  }

  private handleNotification(notification: any): void {
    // Add notification to alerts
    const alert: Alert = {
      id: Date.now().toString(),
      type: notification.data?.type || 'info',
      title: notification.title || 'Energy Copilot',
      message: notification.message || notification.body,
      timestamp: new Date(),
      isRead: false,
    };

    this.addAlert(alert);
  }

  async addAlert(alert: Alert): Promise<void> {
    this.alerts.unshift(alert);
    
    // Keep only last 50 alerts
    if (this.alerts.length > 50) {
      this.alerts = this.alerts.slice(0, 50);
    }

    await this.saveAlerts();
  }

  async getRecentAlerts(limit: number = 10): Promise<Alert[]> {
    return this.alerts.slice(0, limit);
  }

  async getAllAlerts(): Promise<Alert[]> {
    return [...this.alerts];
  }

  async markAlertAsRead(alertId: string): Promise<void> {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.isRead = true;
      await this.saveAlerts();
    }
  }

  async markAllAlertsAsRead(): Promise<void> {
    this.alerts.forEach(alert => alert.isRead = true);
    await this.saveAlerts();
  }

  async clearAlerts(): Promise<void> {
    this.alerts = [];
    await this.saveAlerts();
  }

  async sendLocalNotification(title: string, message: string, data?: any): Promise<void> {
    PushNotification.localNotification({
      title,
      message,
      data: data || {},
      playSound: true,
      soundName: 'default',
    });
  }

  async scheduleNotification(title: string, message: string, date: Date, data?: any): Promise<void> {
    PushNotification.localNotificationSchedule({
      title,
      message,
      date,
      data: data || {},
      playSound: true,
      soundName: 'default',
    });
  }

  async cancelAllNotifications(): Promise<void> {
    PushNotification.cancelAllLocalNotifications();
  }

  // Energy-specific notification methods
  async notifyHighConsumption(consumption: number, threshold: number): Promise<void> {
    const alert: Alert = {
      id: Date.now().toString(),
      type: 'warning',
      title: 'High Energy Consumption',
      message: `Current consumption (${consumption.toFixed(1)} kW) exceeds threshold (${threshold.toFixed(1)} kW)`,
      timestamp: new Date(),
      isRead: false,
    };

    await this.addAlert(alert);
    await this.sendLocalNotification(alert.title, alert.message);
  }

  async notifyDeviceOffline(deviceName: string): Promise<void> {
    const alert: Alert = {
      id: Date.now().toString(),
      type: 'warning',
      title: 'Device Offline',
      message: `${deviceName} has gone offline`,
      timestamp: new Date(),
      isRead: false,
    };

    await this.addAlert(alert);
    await this.sendLocalNotification(alert.title, alert.message);
  }

  async notifyEnergySavings(savings: number): Promise<void> {
    const alert: Alert = {
      id: Date.now().toString(),
      type: 'success',
      title: 'Energy Savings',
      message: `You've saved $${savings.toFixed(2)} this month through optimization!`,
      timestamp: new Date(),
      isRead: false,
    };

    await this.addAlert(alert);
    await this.sendLocalNotification(alert.title, alert.message);
  }

  async notifyAnomalyDetected(anomaly: string): Promise<void> {
    const alert: Alert = {
      id: Date.now().toString(),
      type: 'warning',
      title: 'Anomaly Detected',
      message: anomaly,
      timestamp: new Date(),
      isRead: false,
    };

    await this.addAlert(alert);
    await this.sendLocalNotification(alert.title, alert.message);
  }

  async notifyRecommendation(recommendation: string): Promise<void> {
    const alert: Alert = {
      id: Date.now().toString(),
      type: 'info',
      title: 'Energy Recommendation',
      message: recommendation,
      timestamp: new Date(),
      isRead: false,
    };

    await this.addAlert(alert);
  }
}

export default new NotificationService();