import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  CircularProgress,
  Tooltip,
  Badge,
  Divider,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Notifications as NotificationsIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  DoneAll as DoneAllIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import axios from 'axios';

interface EnergyAlert {
  alert_id: string;
  alert_type: string;
  severity: string;
  status: string;
  title: string;
  message: string;
  timestamp: string;
  device_id?: string;
  location?: string;
  value?: number;
  threshold?: number;
  metadata: Record<string, unknown>;
  acknowledged_at?: string;
  resolved_at?: string;
}

interface AlertSummary {
  total_alerts: number;
  active_alerts: number;
  acknowledged_alerts: number;
  resolved_alerts: number;
  severity_breakdown: Record<string, number>;
  type_breakdown: Record<string, number>;
  timestamp: string;
}

const AlertsWidget: React.FC = () => {
  const [alerts, setAlerts] = useState<EnergyAlert[]>([]);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);

      const [alertsResponse, summaryResponse] = await Promise.all([
        axios.get('/api/alerts/active'),
        axios.get('/api/alerts/summary'),
      ]);

      setAlerts(alertsResponse.data.alerts || []);
      setSummary(summaryResponse.data);
    } catch (err) {
      console.error('Error fetching alerts:', err);
      setError('Failed to load alerts');
      // Set mock data for demo purposes
      setAlerts([
        {
          alert_id: 'demo_1',
          alert_type: 'high_consumption',
          severity: 'warning',
          status: 'active',
          title: 'High Energy Consumption',
          message: 'Current consumption (85 kWh) exceeds normal threshold (100 kWh)',
          timestamp: new Date().toISOString(),
          value: 85,
          threshold: 100,
          metadata: {},
        },
        {
          alert_id: 'demo_2',
          alert_type: 'budget_warning',
          severity: 'info',
          status: 'active',
          title: 'Budget Warning',
          message: "You've used 80% of your monthly energy budget",
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          metadata: { budget_percent: 80 },
        },
      ]);
      setSummary({
        total_alerts: 5,
        active_alerts: 2,
        acknowledged_alerts: 1,
        resolved_alerts: 2,
        severity_breakdown: { critical: 0, warning: 1, info: 1 },
        type_breakdown: { high_consumption: 1, budget_warning: 1 },
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    // Refresh alerts every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAcknowledge = async (alertId: string) => {
    try {
      await axios.post(`/api/alerts/${alertId}/acknowledge`);
      fetchAlerts();
    } catch (err) {
      console.error('Error acknowledging alert:', err);
    }
  };

  const handleDismiss = async (alertId: string) => {
    try {
      await axios.post(`/api/alerts/${alertId}/dismiss`);
      fetchAlerts();
    } catch (err) {
      console.error('Error dismissing alert:', err);
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      default:
        return <InfoIcon />;
    }
  };

  const getSeverityColor = (severity: string): 'error' | 'warning' | 'info' | 'success' | 'default' => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const formatAlertType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const criticalCount = summary?.severity_breakdown?.critical || 0;
  const warningCount = summary?.severity_breakdown?.warning || 0;
  const activeCount = summary?.active_alerts || alerts.length;

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <Badge badgeContent={activeCount} color="error">
              <NotificationsIcon color="primary" />
            </Badge>
            <Typography variant="h6">Energy Alerts</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <IconButton size="small" onClick={fetchAlerts} disabled={loading}>
              <RefreshIcon fontSize="small" />
            </IconButton>
            <IconButton size="small" onClick={() => setExpanded(!expanded)}>
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
        </Box>

        {/* Summary Stats */}
        <Box display="flex" gap={1} mb={2} flexWrap="wrap">
          {criticalCount > 0 && (
            <Chip
              icon={<ErrorIcon />}
              label={`${criticalCount} Critical`}
              size="small"
              color="error"
              variant="outlined"
            />
          )}
          {warningCount > 0 && (
            <Chip
              icon={<WarningIcon />}
              label={`${warningCount} Warning`}
              size="small"
              color="warning"
              variant="outlined"
            />
          )}
          {activeCount === 0 && (
            <Chip
              icon={<CheckCircleIcon />}
              label="All Clear"
              size="small"
              color="success"
              variant="outlined"
            />
          )}
        </Box>

        {/* Loading State */}
        {loading && (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress size={32} />
          </Box>
        )}

        {/* Error State */}
        {error && !loading && (
          <Typography color="text.secondary" textAlign="center" py={2}>
            {error}
          </Typography>
        )}

        {/* Alerts List */}
        {!loading && alerts.length > 0 && (
          <List sx={{ flexGrow: 1, overflow: 'auto', maxHeight: expanded ? 400 : 200 }}>
            {alerts.slice(0, expanded ? undefined : 3).map((alert, index) => (
              <React.Fragment key={alert.alert_id}>
                <ListItem
                  sx={{
                    borderRadius: 1,
                    mb: 1,
                    bgcolor: 'background.paper',
                    border: 1,
                    borderColor: 'divider',
                    '&:hover': {
                      bgcolor: 'action.hover',
                    },
                  }}
                  secondaryAction={
                    <Box>
                      <Tooltip title="Acknowledge">
                        <IconButton
                          size="small"
                          onClick={() => handleAcknowledge(alert.alert_id)}
                        >
                          <DoneAllIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Dismiss">
                        <IconButton
                          size="small"
                          onClick={() => handleDismiss(alert.alert_id)}
                        >
                          <CloseIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  }
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {getSeverityIcon(alert.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body2" fontWeight="medium">
                          {alert.title}
                        </Typography>
                        <Chip
                          label={formatAlertType(alert.alert_type)}
                          size="small"
                          color={getSeverityColor(alert.severity)}
                          sx={{ height: 20, fontSize: '0.7rem' }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary" display="block">
                          {alert.message}
                        </Typography>
                        <Typography variant="caption" color="text.disabled">
                          {formatTimestamp(alert.timestamp)}
                          {alert.location && ` • ${alert.location}`}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
                {index < (expanded ? alerts.length - 1 : Math.min(alerts.length, 3) - 1) && (
                  <Divider variant="inset" component="li" />
                )}
              </React.Fragment>
            ))}
          </List>
        )}

        {/* Empty State */}
        {!loading && alerts.length === 0 && !error && (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            py={4}
          >
            <CheckCircleIcon color="success" sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              No active alerts
            </Typography>
            <Typography variant="caption" color="text.disabled">
              Your energy system is running smoothly
            </Typography>
          </Box>
        )}

        {/* Show More */}
        {!loading && alerts.length > 3 && !expanded && (
          <Box textAlign="center" mt={1}>
            <Typography
              variant="caption"
              color="primary"
              sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
              onClick={() => setExpanded(true)}
            >
              Show {alerts.length - 3} more alerts
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AlertsWidget;
