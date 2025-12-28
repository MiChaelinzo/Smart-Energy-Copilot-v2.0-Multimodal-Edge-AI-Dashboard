/**
 * Smart Energy Copilot Mobile App
 * React Native application for energy management and smart home control
 */

import React, { useEffect, useState } from 'react';
import {
  SafeAreaProvider,
  SafeAreaView,
} from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import {
  StatusBar,
  StyleSheet,
  Alert,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { enableScreens } from 'react-native-screens';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Screens
import DashboardScreen from './screens/DashboardScreen';
import DevicesScreen from './screens/DevicesScreen';
import EnergyScreen from './screens/EnergyScreen';
import RecommendationsScreen from './screens/RecommendationsScreen';
import SettingsScreen from './screens/SettingsScreen';
import LoginScreen from './screens/LoginScreen';
import CameraScreen from './screens/CameraScreen';
import VoiceControlScreen from './screens/VoiceControlScreen';
import ForecastScreen from './screens/ForecastScreen';
import AutomationScreen from './screens/AutomationScreen';

// Services
import { EnergyService } from './services/EnergyService';
import { DeviceService } from './services/DeviceService';
import { NotificationService } from './services/NotificationService';
import { VoiceService } from './services/VoiceService';

// Context
import { AppProvider } from './context/AppContext';

// Types
interface User {
  id: string;
  name: string;
  email: string;
  preferences: {
    theme: 'light' | 'dark';
    notifications: boolean;
    voiceEnabled: boolean;
  };
}

// Enable screens for better performance
enableScreens();

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

const TabNavigator = () => (
  <Tab.Navigator
    screenOptions={({ route }) => ({
      tabBarIcon: ({ focused, color, size }) => {
        let iconName: string;

        switch (route.name) {
          case 'Dashboard':
            iconName = 'dashboard';
            break;
          case 'Energy':
            iconName = 'flash-on';
            break;
          case 'Devices':
            iconName = 'devices';
            break;
          case 'Recommendations':
            iconName = 'lightbulb';
            break;
          case 'Settings':
            iconName = 'settings';
            break;
          default:
            iconName = 'help';
        }

        return <Icon name={iconName} size={size} color={color} />;
      },
      tabBarActiveTintColor: '#2196F3',
      tabBarInactiveTintColor: 'gray',
      headerShown: false,
    })}
  >
    <Tab.Screen name="Dashboard" component={DashboardScreen} />
    <Tab.Screen name="Energy" component={EnergyScreen} />
    <Tab.Screen name="Devices" component={DevicesScreen} />
    <Tab.Screen name="Recommendations" component={RecommendationsScreen} />
    <Tab.Screen name="Settings" component={SettingsScreen} />
  </Tab.Navigator>
);

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Request permissions
      await requestPermissions();

      // Initialize services
      await initializeServices();

      // Check authentication
      const authToken = await AsyncStorage.getItem('authToken');
      if (authToken) {
        // Validate token and get user info
        const userData = await validateAuthToken(authToken);
        if (userData) {
          setUser(userData);
          setIsAuthenticated(true);
        }
      }

      setIsLoading(false);
    } catch (error) {
      console.error('App initialization error:', error);
      setIsLoading(false);
    }
  };

  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      try {
        const permissions = [
          PermissionsAndroid.PERMISSIONS.CAMERA,
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
          PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
        ];

        const granted = await PermissionsAndroid.requestMultiple(permissions);
        
        const allGranted = Object.values(granted).every(
          permission => permission === PermissionsAndroid.RESULTS.GRANTED
        );

        if (!allGranted) {
          Alert.alert(
            'Permissions Required',
            'Some features may not work without the required permissions.',
            [{ text: 'OK' }]
          );
        }
      } catch (error) {
        console.error('Permission request error:', error);
      }
    }
  };

  const initializeServices = async () => {
    try {
      // Initialize notification service
      await NotificationService.initialize();

      // Initialize voice service
      await VoiceService.initialize();

      // Initialize energy service
      await EnergyService.initialize();

      // Initialize device service
      await DeviceService.initialize();

      console.log('All services initialized successfully');
    } catch (error) {
      console.error('Service initialization error:', error);
    }
  };

  const validateAuthToken = async (token: string): Promise<User | null> => {
    try {
      // Mock authentication validation
      // In real implementation, this would validate with backend
      const mockUser: User = {
        id: 'user_123',
        name: 'John Doe',
        email: 'john.doe@example.com',
        preferences: {
          theme: 'light',
          notifications: true,
          voiceEnabled: true,
        },
      };

      return mockUser;
    } catch (error) {
      console.error('Auth validation error:', error);
      return null;
    }
  };

  const handleLogin = async (credentials: { email: string; password: string }) => {
    try {
      // Mock login process
      // In real implementation, this would authenticate with backend
      const authToken = 'mock_auth_token_' + Date.now();
      await AsyncStorage.setItem('authToken', authToken);

      const userData = await validateAuthToken(authToken);
      if (userData) {
        setUser(userData);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Login Failed', 'Please check your credentials and try again.');
    }
  };

  const handleLogout = async () => {
    try {
      await AsyncStorage.removeItem('authToken');
      setUser(null);
      setIsAuthenticated(false);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  if (isLoading) {
    // Show loading screen
    return (
      <SafeAreaProvider>
        <SafeAreaView style={styles.container}>
          <StatusBar barStyle="dark-content" backgroundColor="#ffffff" />
          {/* Loading component would go here */}
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <AppProvider user={user} onLogout={handleLogout}>
        <NavigationContainer>
          <StatusBar barStyle="dark-content" backgroundColor="#ffffff" />
          
          {isAuthenticated ? (
            <Stack.Navigator screenOptions={{ headerShown: false }}>
              <Stack.Screen name="Main" component={TabNavigator} />
              <Stack.Screen 
                name="Camera" 
                component={CameraScreen}
                options={{ headerShown: true, title: 'Scan Document' }}
              />
              <Stack.Screen 
                name="VoiceControl" 
                component={VoiceControlScreen}
                options={{ headerShown: true, title: 'Voice Control' }}
              />
              <Stack.Screen 
                name="Forecast" 
                component={ForecastScreen}
                options={{ headerShown: true, title: 'Energy Forecast' }}
              />
              <Stack.Screen 
                name="Automation" 
                component={AutomationScreen}
                options={{ headerShown: true, title: 'Automation Rules' }}
              />
            </Stack.Navigator>
          ) : (
            <LoginScreen onLogin={handleLogin} />
          )}
        </NavigationContainer>
      </AppProvider>
    </SafeAreaProvider>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
});

export default App;