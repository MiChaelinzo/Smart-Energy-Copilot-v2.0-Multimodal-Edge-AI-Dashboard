/**
 * App Context for global state management
 */

import React, { createContext, useContext, useReducer, ReactNode } from 'react';

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

interface AppState {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  notifications: any[];
}

type AppAction =
  | { type: 'SET_USER'; payload: User | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CONNECTION_STATUS'; payload: 'connected' | 'disconnected' | 'connecting' }
  | { type: 'ADD_NOTIFICATION'; payload: any }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS' };

// Initial state
const initialState: AppState = {
  user: null,
  isLoading: false,
  error: null,
  connectionStatus: 'disconnected',
  notifications: [],
};

// Reducer
const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_USER':
      return { ...state, user: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_CONNECTION_STATUS':
      return { ...state, connectionStatus: action.payload };
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [action.payload, ...state.notifications.slice(0, 49)], // Keep last 50
      };
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload),
      };
    case 'CLEAR_NOTIFICATIONS':
      return { ...state, notifications: [] };
    default:
      return state;
  }
};

// Context
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  actions: {
    setUser: (user: User | null) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    setConnectionStatus: (status: 'connected' | 'disconnected' | 'connecting') => void;
    addNotification: (notification: any) => void;
    removeNotification: (id: string) => void;
    clearNotifications: () => void;
  };
} | undefined>(undefined);

// Provider component
interface AppProviderProps {
  children: ReactNode;
  user?: User | null;
  onLogout?: () => void;
}

export const AppProvider: React.FC<AppProviderProps> = ({ 
  children, 
  user = null,
  onLogout 
}) => {
  const [state, dispatch] = useReducer(appReducer, {
    ...initialState,
    user,
  });

  const actions = {
    setUser: (user: User | null) => dispatch({ type: 'SET_USER', payload: user }),
    setLoading: (loading: boolean) => dispatch({ type: 'SET_LOADING', payload: loading }),
    setError: (error: string | null) => dispatch({ type: 'SET_ERROR', payload: error }),
    setConnectionStatus: (status: 'connected' | 'disconnected' | 'connecting') =>
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: status }),
    addNotification: (notification: any) =>
      dispatch({ type: 'ADD_NOTIFICATION', payload: notification }),
    removeNotification: (id: string) =>
      dispatch({ type: 'REMOVE_NOTIFICATION', payload: id }),
    clearNotifications: () => dispatch({ type: 'CLEAR_NOTIFICATIONS' }),
  };

  return (
    <AppContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </AppContext.Provider>
  );
};

// Hook to use the context
export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

export default AppContext;