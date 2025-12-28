/**
 * Dashboard Screen - Main overview of energy consumption and smart home status
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Dimensions,
  Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { LineChart, PieChart } from 'react-native-chart-kit';
import LinearGradient from 'react-native-linear-gradient';

// Services
import { EnergyService } from '../services/EnergyService';
import { DeviceService } from '../services/DeviceService';
import { NotificationService } from '../services/NotificationService';

// Components
import EnergyOverviewCard from '../components/EnergyOverviewCard';
import QuickActionsPanel from '../components/QuickActionsPanel';
import RecentRecommendations from '../components/RecentRecommendations';
import DeviceStatusGrid from '../components/DeviceStatusGrid';

// Types
interface DashboardData {
  currentConsumption: number;
  dailyConsumption: number;
  monthlyConsumption: number;
  estimatedCost: number;
  savingsThisMonth: number;
  activeDevices: number;
  totalDevices: number;
  recentConsumption: number[];
  consumptionByCategory: Array<{
    name: string;
    consumption: number;
    color: string;
  }>;
  alerts: Array<{
    id: string;
    type: 'warning' | 'info' | 'success';
    message: string;
    timestamp: Date;
  }>;
}

const { width: screenWidth } = Dimensions.get('window');

const DashboardScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  useFocusEffect(
    useCallback(() => {
      loadDashboardData();
    }, [])
  );

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);

      // Load energy data
      const energyData = await EnergyService.getCurrentStatus();
      const consumptionHistory = await EnergyService.getConsumptionHistory(7); // Last 7 days
      const categoryBreakdown = await EnergyService.getConsumptionByCategory();

      // Load device data
      const deviceStatus = await DeviceService.getDeviceStatus();

      // Load alerts
      const alerts = await NotificationService.getRecentAlerts();

      // Combine data
      const data: DashboardData = {
        currentConsumption: energyData.currentConsumption || 2.5,
        dailyConsumption: energyData.dailyConsumption || 45.2,
        monthlyConsumption: energyData.monthlyConsumption || 1250,
        estimatedCost: energyData.estimatedCost || 125.50,
        savingsThisMonth: energyData.savingsThisMonth || 23.75,
        activeDevices: deviceStatus.activeDevices || 12,
        totalDevices: deviceStatus.totalDevices || 15,
        recentConsumption: consumptionHistory || [42, 38, 45, 52, 48, 41, 45],
        consumptionByCategory: categoryBreakdown || [
          { name: 'HVAC', consumption: 45, color: '#FF6B6B' },
          { name: 'Lighting', consumption: 20, color: '#4ECDC4' },
          { name: 'Appliances', consumption: 25, color: '#45B7D1' },
          { name: 'Electronics', consumption: 10, color: '#96CEB4' },
        ],
        alerts: alerts || [],
      };

      setDashboardData(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      Alert.alert('Error', 'Failed to load dashboard data. Please try again.');
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadDashboardData();
  };

  const handleQuickAction = (action: string) => {
    switch (action) {
      case 'scan_bill':
        navigation.navigate('Camera');
        break;
      case 'voice_control':
        navigation.navigate('VoiceControl');
        break;
      case 'view_forecast':
        navigation.navigate('Forecast');
        break;
      case 'manage_automation':
        navigation.navigate('Automation');
        break;
      default:
        console.log('Unknown action:', action);
    }
  };

  const renderEnergyChart = () => {
    if (!dashboardData) return null;

    const chartConfig = {
      backgroundColor: '#ffffff',
      backgroundGradientFrom: '#ffffff',
      backgroundGradientTo: '#ffffff',
      decimalPlaces: 1,
      color: (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
      labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
      style: {
        borderRadius: 16,
      },
      propsForDots: {
        r: '6',
        strokeWidth: '2',
        stroke: '#2196F3',
      },
    };

    const data = {
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [
        {
          data: dashboardData.recentConsumption,
          color: (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
          strokeWidth: 2,
        },
      ],
    };

    return (
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Weekly Consumption (kWh)</Text>
        <LineChart
          data={data}
          width={screenWidth - 40}
          height={200}
          chartConfig={chartConfig}
          bezier
          style={styles.chart}
        />
      </View>
    );
  };

  const renderConsumptionBreakdown = () => {
    if (!dashboardData) return null;

    const pieData = dashboardData.consumptionByCategory.map((item, index) => ({
      name: item.name,
      population: item.consumption,
      color: item.color,
      legendFontColor: '#7F7F7F',
      legendFontSize: 12,
    }));

    return (
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Consumption by Category</Text>
        <PieChart
          data={pieData}
          width={screenWidth - 40}
          height={200}
          chartConfig={{
            color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
          }}
          accessor="population"
          backgroundColor="transparent"
          paddingLeft="15"
          absolute
        />
      </View>
    );
  };

  const renderAlerts = () => {
    if (!dashboardData || dashboardData.alerts.length === 0) return null;

    return (
      <View style={styles.alertsContainer}>
        <Text style={styles.sectionTitle}>Recent Alerts</Text>
        {dashboardData.alerts.slice(0, 3).map((alert) => (
          <View key={alert.id} style={[styles.alertItem, styles[`alert${alert.type}`]]}>
            <Icon
              name={
                alert.type === 'warning' ? 'warning' :
                alert.type === 'success' ? 'check-circle' : 'info'
              }
              size={20}
              color={
                alert.type === 'warning' ? '#FF9800' :
                alert.type === 'success' ? '#4CAF50' : '#2196F3'
              }
            />
            <Text style={styles.alertText}>{alert.message}</Text>
          </View>
        ))}
      </View>
    );
  };

  if (isLoading && !dashboardData) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading dashboard...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <LinearGradient
        colors={['#2196F3', '#21CBF3']}
        style={styles.header}
      >
        <Text style={styles.headerTitle}>Energy Dashboard</Text>
        <Text style={styles.headerSubtitle}>
          Last updated: {lastUpdated.toLocaleTimeString()}
        </Text>
      </LinearGradient>

      {dashboardData && (
        <>
          <EnergyOverviewCard
            currentConsumption={dashboardData.currentConsumption}
            dailyConsumption={dashboardData.dailyConsumption}
            monthlyConsumption={dashboardData.monthlyConsumption}
            estimatedCost={dashboardData.estimatedCost}
            savingsThisMonth={dashboardData.savingsThisMonth}
          />

          <QuickActionsPanel onAction={handleQuickAction} />

          <DeviceStatusGrid
            activeDevices={dashboardData.activeDevices}
            totalDevices={dashboardData.totalDevices}
            onDevicePress={() => navigation.navigate('Devices')}
          />

          {renderEnergyChart()}

          {renderConsumptionBreakdown()}

          <RecentRecommendations
            onViewAll={() => navigation.navigate('Recommendations')}
          />

          {renderAlerts()}
        </>
      )}

      <View style={styles.bottomSpacing} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    paddingTop: 40,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#ffffff',
    opacity: 0.8,
  },
  chartContainer: {
    backgroundColor: '#ffffff',
    margin: 10,
    padding: 15,
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  alertsContainer: {
    backgroundColor: '#ffffff',
    margin: 10,
    padding: 15,
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    marginBottom: 8,
    borderRadius: 8,
    borderLeftWidth: 4,
  },
  alertwarning: {
    backgroundColor: '#FFF3E0',
    borderLeftColor: '#FF9800',
  },
  alertinfo: {
    backgroundColor: '#E3F2FD',
    borderLeftColor: '#2196F3',
  },
  alertsuccess: {
    backgroundColor: '#E8F5E8',
    borderLeftColor: '#4CAF50',
  },
  alertText: {
    flex: 1,
    marginLeft: 10,
    fontSize: 14,
    color: '#333',
  },
  bottomSpacing: {
    height: 20,
  },
});

export default DashboardScreen;