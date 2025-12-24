import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Avatar,
  Tooltip,
  Menu,
  MenuItem,
  Badge,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  DeviceThermostat as ThermostatIcon,
  ElectricMeter as MeterIcon,
  SolarPower as SolarIcon,
  Battery3Bar as BatteryIcon,
  Devices as DevicesIcon,
  Circle as CircleIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useDashboard } from '../../contexts/DashboardContext';

const DeviceStatusWidget: React.FC = () => {
  const { state } = useDashboard();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedDevice, setSelectedDevice] = useState<any>(null);

  const getDeviceIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'thermostat':
        return <ThermostatIcon />;
      case 'smart_meter':
        return <MeterIcon />;
      case 'solar_panel':
        return <SolarIcon />;
      case 'battery':
        return <BatteryIcon />;
      default:
        return <DevicesIcon />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'online':
        return 'success';
      case 'offline':
        return 'error';
      case 'error':
        return 'error';
      case 'maintenance':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    const color = getStatusColor(status);
    return (
      <CircleIcon
        sx={{
          fontSize: 12,
          color: color === 'success' ? 'success.main' :
                 color === 'error' ? 'error.main' :
                 color === 'warning' ? 'warning.main' : 'text.disabled'
        }}
      />
    );
  };

  const formatConsumption = (consumption: number) => {
    if (consumption < 0) {
      return `+${Math.abs(consumption).toFixed(1)} kW`; // Generation
    }
    return `${consumption.toFixed(1)} kW`;
  };

  const getConsumptionColor = (consumption: number) => {
    if (consumption < 0) return 'success.main'; // Generation (green)
    if (consumption > 10) return 'error.main'; // High consumption (red)
    if (consumption > 5) return 'warning.main'; // Medium consumption (orange)
    return 'text.primary'; // Low consumption (default)
  };

  const handleDeviceMenuOpen = (event: React.MouseEvent<HTMLElement>, device: any) => {
    setAnchorEl(event.currentTarget);
    setSelectedDevice(device);
  };

  const handleDeviceMenuClose = () => {
    setAnchorEl(null);
    setSelectedDevice(null);
  };

  const getDeviceStats = () => {
    const total = state.devices.length;
    const online = state.devices.filter(d => d.status === 'online').length;
    const offline = state.devices.filter(d => d.status === 'offline').length;
    const errors = state.devices.filter(d => d.status === 'error').length;

    return { total, online, offline, errors };
  };

  const stats = getDeviceStats();

  return (
    <>
      <Card className="widget-container" sx={{ height: '100%' }}>
        <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Widget Header */}
          <Box className="widget-header">
            <Typography variant="h6" component="h2">
              Device Status
            </Typography>
            <IconButton size="small">
              <MoreVertIcon />
            </IconButton>
          </Box>

          {/* Status Summary */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={`${stats.online} Online`}
                size="small"
                color="success"
                variant="outlined"
              />
              {stats.offline > 0 && (
                <Chip
                  label={`${stats.offline} Offline`}
                  size="small"
                  color="error"
                  variant="outlined"
                />
              )}
              {stats.errors > 0 && (
                <Chip
                  label={`${stats.errors} Errors`}
                  size="small"
                  color="error"
                  variant="filled"
                />
              )}
            </Box>
          </Box>

          {/* Device List */}
          <Box className="widget-content" sx={{ flex: 1, overflow: 'auto' }}>
            {state.devices.length > 0 ? (
              <List dense>
                {state.devices.map((device) => (
                  <ListItem
                    key={device.id}
                    sx={{
                      mb: 1,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <ListItemIcon>
                      <Badge
                        overlap="circular"
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                        badgeContent={getStatusIcon(device.status)}
                      >
                        <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                          {getDeviceIcon(device.type)}
                        </Avatar>
                      </Badge>
                    </ListItemIcon>

                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle2">
                            {device.name}
                          </Typography>
                          <Chip
                            label={device.type.replace('_', ' ')}
                            size="small"
                            variant="outlined"
                            sx={{ textTransform: 'capitalize' }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Typography
                              variant="body2"
                              sx={{ color: getConsumptionColor(device.current_consumption) }}
                            >
                              {formatConsumption(device.current_consumption)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Efficiency: {(device.efficiency_rating * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            Last update: {new Date(device.last_update).toLocaleTimeString()}
                          </Typography>
                        </Box>
                      }
                    />

                    <ListItemSecondaryAction>
                      <Tooltip title="Device Options">
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={(e) => handleDeviceMenuOpen(e, device)}
                        >
                          <MoreVertIcon />
                        </IconButton>
                      </Tooltip>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  color: 'text.secondary',
                }}
              >
                <Typography variant="body1">
                  {state.loading ? 'Loading devices...' : 'No devices connected'}
                </Typography>
              </Box>
            )}

            {/* Total Consumption Summary */}
            {state.devices.length > 0 && (
              <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                <Typography variant="caption" color="text.secondary" gutterBottom>
                  Total Current Consumption
                </Typography>
                <Typography variant="h6" color="primary">
                  {state.devices.reduce((sum, device) => sum + Math.max(0, device.current_consumption), 0).toFixed(1)} kW
                </Typography>
                {state.devices.some(d => d.current_consumption < 0) && (
                  <Typography variant="body2" color="success.main">
                    Generation: {Math.abs(state.devices.reduce((sum, device) => sum + Math.min(0, device.current_consumption), 0)).toFixed(1)} kW
                  </Typography>
                )}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Device Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleDeviceMenuClose}
      >
        <MenuItem onClick={handleDeviceMenuClose}>
          <ListItemIcon>
            <InfoIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Device Details</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDeviceMenuClose}>
          <ListItemIcon>
            <HistoryIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>View History</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDeviceMenuClose}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
};

export default DeviceStatusWidget;