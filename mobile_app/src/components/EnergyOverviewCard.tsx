/**
 * Energy Overview Card Component
 * Displays current energy consumption and cost information
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';

interface EnergyOverviewCardProps {
  currentConsumption: number;
  dailyConsumption: number;
  monthlyConsumption: number;
  estimatedCost: number;
  savingsThisMonth: number;
}

const EnergyOverviewCard: React.FC<EnergyOverviewCardProps> = ({
  currentConsumption,
  dailyConsumption,
  monthlyConsumption,
  estimatedCost,
  savingsThisMonth,
}) => {
  return (
    <LinearGradient
      colors={['#4CAF50', '#45A049']}
      style={styles.container}
    >
      <View style={styles.header}>
        <Icon name="flash-on" size={24} color="#ffffff" />
        <Text style={styles.headerTitle}>Energy Overview</Text>
      </View>

      <View style={styles.content}>
        <View style={styles.mainMetric}>
          <Text style={styles.mainValue}>{currentConsumption.toFixed(1)}</Text>
          <Text style={styles.mainUnit}>kW</Text>
          <Text style={styles.mainLabel}>Current Usage</Text>
        </View>

        <View style={styles.metricsGrid}>
          <View style={styles.metric}>
            <Text style={styles.metricValue}>{dailyConsumption.toFixed(1)}</Text>
            <Text style={styles.metricUnit}>kWh</Text>
            <Text style={styles.metricLabel}>Today</Text>
          </View>

          <View style={styles.metric}>
            <Text style={styles.metricValue}>{monthlyConsumption.toFixed(0)}</Text>
            <Text style={styles.metricUnit}>kWh</Text>
            <Text style={styles.metricLabel}>This Month</Text>
          </View>

          <View style={styles.metric}>
            <Text style={styles.metricValue}>${estimatedCost.toFixed(0)}</Text>
            <Text style={styles.metricUnit}>USD</Text>
            <Text style={styles.metricLabel}>Est. Cost</Text>
          </View>

          <View style={styles.metric}>
            <Text style={[styles.metricValue, styles.savingsValue]}>
              ${savingsThisMonth.toFixed(0)}
            </Text>
            <Text style={styles.metricUnit}>USD</Text>
            <Text style={styles.metricLabel}>Saved</Text>
          </View>
        </View>
      </View>

      <TouchableOpacity style={styles.detailsButton}>
        <Text style={styles.detailsButtonText}>View Details</Text>
        <Icon name="arrow-forward" size={16} color="#ffffff" />
      </TouchableOpacity>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 15,
    borderRadius: 16,
    padding: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
    marginLeft: 8,
  },
  content: {
    marginBottom: 20,
  },
  mainMetric: {
    alignItems: 'center',
    marginBottom: 20,
  },
  mainValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#ffffff',
    lineHeight: 56,
  },
  mainUnit: {
    fontSize: 16,
    color: '#ffffff',
    opacity: 0.9,
    marginTop: -8,
  },
  mainLabel: {
    fontSize: 14,
    color: '#ffffff',
    opacity: 0.8,
    marginTop: 4,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metric: {
    width: '48%',
    alignItems: 'center',
    marginBottom: 15,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: '600',
    color: '#ffffff',
  },
  metricUnit: {
    fontSize: 12,
    color: '#ffffff',
    opacity: 0.8,
  },
  metricLabel: {
    fontSize: 12,
    color: '#ffffff',
    opacity: 0.7,
    marginTop: 2,
  },
  savingsValue: {
    color: '#FFD700', // Gold color for savings
  },
  detailsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  detailsButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
    marginRight: 8,
  },
});

export default EnergyOverviewCard;