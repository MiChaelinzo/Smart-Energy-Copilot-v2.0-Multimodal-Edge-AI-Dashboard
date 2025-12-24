import React, { createContext, useContext, useReducer, ReactNode, useEffect } from 'react';
import { useWebSocket } from './WebSocketContext';

interface EnergyData {
  id: string;
  timestamp: string;
  consumption_kwh: number;
  cost_usd: number;
  source: string;
  confidence_score: number;
}

interface Recommendation {
  id: string;
  type: string;
  priority: string;
  title: string;
  description: string;
  implementation_steps: string[];
  estimated_savings: {
    annual_cost_usd: number;
    annual_kwh: number;
    co2_reduction_kg: number;
  };
  difficulty: string;
  agent_source: string;
  confidence: number;
  created_at: string;
  status: string;
}

interface Device {
  id: string;
  name: string;
  type: string;
  status: string;
  current_consumption: number;
  efficiency_rating: number;
  last_update: string;
}

interface DashboardState {
  energyData: EnergyData[];
  recommendations: Recommendation[];
  devices: Device[];
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

type DashboardAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_ENERGY_DATA'; payload: EnergyData[] }
  | { type: 'SET_RECOMMENDATIONS'; payload: Recommendation[] }
  | { type: 'SET_DEVICES'; payload: Device[] }
  | { type: 'UPDATE_RECOMMENDATION'; payload: { id: string; status: string } }
  | { type: 'ADD_ENERGY_DATA'; payload: EnergyData }
  | { type: 'SET_LAST_UPDATE'; payload: string };

const initialState: DashboardState = {
  energyData: [],
  recommendations: [],
  devices: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

function dashboardReducer(state: DashboardState, action: DashboardAction): DashboardState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_ENERGY_DATA':
      return { ...state, energyData: action.payload, loading: false };
    case 'SET_RECOMMENDATIONS':
      return { ...state, recommendations: action.payload, loading: false };
    case 'SET_DEVICES':
      return { ...state, devices: action.payload, loading: false };
    case 'UPDATE_RECOMMENDATION':
      return {
        ...state,
        recommendations: state.recommendations.map(rec =>
          rec.id === action.payload.id
            ? { ...rec, status: action.payload.status }
            : rec
        ),
      };
    case 'ADD_ENERGY_DATA':
      return {
        ...state,
        energyData: [action.payload, ...state.energyData.slice(0, 99)], // Keep last 100 records
      };
    case 'SET_LAST_UPDATE':
      return { ...state, lastUpdate: action.payload };
    default:
      return state;
  }
}

interface DashboardContextType {
  state: DashboardState;
  dispatch: React.Dispatch<DashboardAction>;
  refreshData: () => Promise<void>;
  updateRecommendationStatus: (id: string, status: string) => Promise<void>;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

interface DashboardProviderProps {
  children: ReactNode;
}

export const DashboardProvider: React.FC<DashboardProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);
  const { lastMessage } = useWebSocket();

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'real_time_update') {
        dispatch({ type: 'SET_LAST_UPDATE', payload: lastMessage.timestamp });
        
        // Update energy data if available
        if (lastMessage.data?.latest_consumption) {
          const newEnergyData: EnergyData = {
            id: `realtime_${Date.now()}`,
            timestamp: lastMessage.timestamp,
            consumption_kwh: lastMessage.data.latest_consumption.consumption_kwh,
            cost_usd: lastMessage.data.latest_consumption.cost_usd,
            source: 'real_time',
            confidence_score: 1.0,
          };
          dispatch({ type: 'ADD_ENERGY_DATA', payload: newEnergyData });
        }
      } else if (lastMessage.type === 'recommendation_update') {
        dispatch({
          type: 'UPDATE_RECOMMENDATION',
          payload: {
            id: lastMessage.recommendation_id,
            status: lastMessage.status,
          },
        });
      }
    }
  }, [lastMessage]);

  const refreshData = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      // Fetch energy data
      const energyResponse = await fetch('/api/dashboard/energy-data');
      if (!energyResponse.ok) throw new Error('Failed to fetch energy data');
      const energyResult = await energyResponse.json();
      dispatch({ type: 'SET_ENERGY_DATA', payload: energyResult.data });

      // Fetch recommendations
      const recommendationsResponse = await fetch('/api/dashboard/recommendations');
      if (!recommendationsResponse.ok) throw new Error('Failed to fetch recommendations');
      const recommendationsResult = await recommendationsResponse.json();
      dispatch({ type: 'SET_RECOMMENDATIONS', payload: recommendationsResult.recommendations });

      // Mock device data (would come from IoT service in real implementation)
      const mockDevices: Device[] = [
        {
          id: 'device_1',
          name: 'Smart Thermostat',
          type: 'thermostat',
          status: 'online',
          current_consumption: 2.5,
          efficiency_rating: 0.85,
          last_update: new Date().toISOString(),
        },
        {
          id: 'device_2',
          name: 'Smart Meter',
          type: 'smart_meter',
          status: 'online',
          current_consumption: 15.2,
          efficiency_rating: 0.95,
          last_update: new Date().toISOString(),
        },
        {
          id: 'device_3',
          name: 'Solar Panel System',
          type: 'solar_panel',
          status: 'online',
          current_consumption: -8.7, // Negative indicates generation
          efficiency_rating: 0.78,
          last_update: new Date().toISOString(),
        },
      ];
      dispatch({ type: 'SET_DEVICES', payload: mockDevices });

      dispatch({ type: 'SET_LAST_UPDATE', payload: new Date().toISOString() });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  const updateRecommendationStatus = async (id: string, status: string) => {
    try {
      const response = await fetch(`/api/dashboard/recommendations/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status }),
      });

      if (!response.ok) throw new Error('Failed to update recommendation');

      dispatch({ type: 'UPDATE_RECOMMENDATION', payload: { id, status } });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : 'Failed to update recommendation' });
    }
  };

  const value = {
    state,
    dispatch,
    refreshData,
    updateRecommendationStatus,
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};