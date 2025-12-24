import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  AppBar,
  Toolbar,
  IconButton,
  Badge,
  Chip,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  WifiOff as WifiOffIcon,
  Wifi as WifiIcon,
} from '@mui/icons-material';
import { Responsive, WidthProvider } from 'react-grid-layout';
import { useDashboard } from '../contexts/DashboardContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import EnergyOverviewWidget from '../components/widgets/EnergyOverviewWidget';
import ConsumptionChartWidget from '../components/widgets/ConsumptionChartWidget';
import RecommendationsWidget from '../components/widgets/RecommendationsWidget';
import DeviceStatusWidget from '../components/widgets/DeviceStatusWidget';
import CostAnalysisWidget from '../components/widgets/CostAnalysisWidget';

const ResponsiveGridLayout = WidthProvider(Responsive);

const Dashboard: React.FC = () => {
  const { state, refreshData } = useDashboard();
  const { isConnected } = useWebSocket();
  const [showError, setShowError] = useState(false);

  // Default layout configuration
  const defaultLayouts = {
    lg: [
      { i: 'energy-overview', x: 0, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'recommendations', x: 6, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'consumption-chart', x: 0, y: 4, w: 12, h: 6, minW: 8, minH: 4 },
      { i: 'device-status', x: 0, y: 10, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'cost-analysis', x: 6, y: 10, w: 6, h: 4, minW: 4, minH: 3 },
    ],
    md: [
      { i: 'energy-overview', x: 0, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'recommendations', x: 6, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'consumption-chart', x: 0, y: 4, w: 12, h: 6, minW: 6, minH: 4 },
      { i: 'device-status', x: 0, y: 10, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'cost-analysis', x: 6, y: 10, w: 6, h: 4, minW: 4, minH: 3 },
    ],
    sm: [
      { i: 'energy-overview', x: 0, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'recommendations', x: 0, y: 4, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'consumption-chart', x: 0, y: 8, w: 6, h: 6, minW: 4, minH: 4 },
      { i: 'device-status', x: 0, y: 14, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'cost-analysis', x: 0, y: 18, w: 6, h: 4, minW: 4, minH: 3 },
    ],
    xs: [
      { i: 'energy-overview', x: 0, y: 0, w: 4, h: 4, minW: 4, minH: 3 },
      { i: 'recommendations', x: 0, y: 4, w: 4, h: 4, minW: 4, minH: 3 },
      { i: 'consumption-chart', x: 0, y: 8, w: 4, h: 6, minW: 4, minH: 4 },
      { i: 'device-status', x: 0, y: 14, w: 4, h: 4, minW: 4, minH: 3 },
      { i: 'cost-analysis', x: 0, y: 18, w: 4, h: 4, minW: 4, minH: 3 },
    ],
  };

  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 };
  const cols = { lg: 12, md: 12, sm: 6, xs: 4, xxs: 2 };

  useEffect(() => {
    // Load initial data
    refreshData();
  }, []);

  useEffect(() => {
    if (state.error) {
      setShowError(true);
    }
  }, [state.error]);

  const handleRefresh = () => {
    refreshData();
  };

  const handleCloseError = () => {
    setShowError(false);
  };

  const getPendingRecommendationsCount = () => {
    return state.recommendations.filter(rec => rec.status === 'pending').length;
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* App Bar */}
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Smart Energy Copilot v2.0
          </Typography>
          
          {/* Connection Status */}
          <Chip
            icon={isConnected ? <WifiIcon /> : <WifiOffIcon />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
            size="small"
            sx={{ mr: 2, color: 'white', borderColor: 'white' }}
          />
          
          {/* Notifications */}
          <IconButton color="inherit">
            <Badge badgeContent={getPendingRecommendationsCount()} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
          
          {/* Refresh */}
          <IconButton color="inherit" onClick={handleRefresh} disabled={state.loading}>
            <RefreshIcon />
          </IconButton>
          
          {/* Settings */}
          <IconButton color="inherit">
            <SettingsIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth={false} className="dashboard-container">
        {/* Dashboard Header */}
        <Box className="dashboard-header">
          <Typography variant="h4" component="h1" gutterBottom>
            Energy Dashboard
          </Typography>
          {state.lastUpdate && (
            <Typography variant="body2" color="text.secondary">
              Last updated: {new Date(state.lastUpdate).toLocaleString()}
            </Typography>
          )}
        </Box>

        {/* Dashboard Grid */}
        <Box className="dashboard-grid">
          <ResponsiveGridLayout
            className="layout"
            layouts={defaultLayouts}
            breakpoints={breakpoints}
            cols={cols}
            rowHeight={60}
            isDraggable={true}
            isResizable={true}
            margin={[16, 16]}
            containerPadding={[0, 0]}
          >
            {/* Energy Overview Widget */}
            <div key="energy-overview">
              <EnergyOverviewWidget />
            </div>

            {/* Recommendations Widget */}
            <div key="recommendations">
              <RecommendationsWidget />
            </div>

            {/* Consumption Chart Widget */}
            <div key="consumption-chart">
              <ConsumptionChartWidget />
            </div>

            {/* Device Status Widget */}
            <div key="device-status">
              <DeviceStatusWidget />
            </div>

            {/* Cost Analysis Widget */}
            <div key="cost-analysis">
              <CostAnalysisWidget />
            </div>
          </ResponsiveGridLayout>
        </Box>
      </Container>

      {/* Error Snackbar */}
      <Snackbar
        open={showError}
        autoHideDuration={6000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {state.error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Dashboard;